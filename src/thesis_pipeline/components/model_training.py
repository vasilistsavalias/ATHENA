import json
import logging
import re
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from box import ConfigBox
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from torch.optim import AdamW
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

from thesis_pipeline.components.prompt_utils import cap_prompt_to_token_budget
from thesis_pipeline.visualization import ThesisPlotter


class ModelTrainer:
    def __init__(
        self,
        config: ConfigBox,
        hyperparams: ConfigBox,
        output_dir: Path,
        report_dir: Path,
        accelerator: Accelerator,
        *,
        runtime_config: dict | None = None,
        job_name: str | None = None,
    ):
        self.config = config
        self.hyperparams = hyperparams
        self.output_dir = output_dir
        self.logger = logging.getLogger(__name__)

        # Use passed accelerator
        self.accelerator = accelerator
        self.device = self.accelerator.device
        self.runtime_config = runtime_config or {}
        self.job_name = job_name or "stage12_job"

        # Only main process creates plotter
        if self.accelerator.is_main_process:
            self.plotter = ThesisPlotter(report_dir)
        else:
            self.plotter = None

        self.checkpoint_dir = self.output_dir / "_checkpoints"
        self.checkpoint_file = self.checkpoint_dir / "latest.pt"
        self.resume_enabled = bool(self.runtime_config.get("resume_epoch_checkpoints", True))
        self.checkpoint_every_epochs = max(1, int(self.runtime_config.get("checkpoint_every_epochs", 1)))
        self.prompt_token_budget = int(self.runtime_config.get("prompt_token_budget", 77))
        self.log_prompt_truncation = bool(self.runtime_config.get("log_prompt_truncation", True))
        self.conditioning_finetune_mode = str(
            self.runtime_config.get("conditioning_finetune_mode", "full_unet")
        ).strip().lower()
        self.mask_aware_loss = bool(self.runtime_config.get("mask_aware_loss", True))
        self.mask_loss_inside_weight = float(self.runtime_config.get("mask_loss_inside_weight", 2.5))
        self.mask_loss_outside_weight = float(self.runtime_config.get("mask_loss_outside_weight", 0.5))
        self.mask_loss_threshold = float(self.runtime_config.get("mask_loss_threshold", 0.5))
        self.ema_enabled = bool(self.runtime_config.get("ema_enabled", True))
        self.ema_decay = float(self.runtime_config.get("ema_decay", 0.999))
        self.ema_update_after_step = int(self.runtime_config.get("ema_update_after_step", 0))
        self.use_ema_for_validation = bool(self.runtime_config.get("use_ema_for_validation", True))
        self.use_ema_for_saving = bool(self.runtime_config.get("use_ema_for_saving", True))
        self._prompt_seen_total = 0
        self._prompt_truncated_total = 0
        self._prompt_seen_epoch = 0
        self._prompt_truncated_epoch = 0
        self._optimizer_step = 0
        self._ema_state: dict[str, torch.Tensor] = {}

    def _load_pretrained_models(self):
        """Loads all necessary model components from Hugging Face."""
        try:
            mixed_precision = getattr(getattr(self.accelerator, "state", None), "mixed_precision", "no")
            if mixed_precision == "fp16":
                torch_dtype = torch.float16
            elif mixed_precision == "bf16":
                torch_dtype = torch.bfloat16
            else:
                torch_dtype = torch.float32

            model_id = (
                self.hyperparams.get("model_id")
                or self.config.get("model_id")
                or "runwayml/stable-diffusion-inpainting"
            )

            # Ensure only the main process downloads the model first.
            with self.accelerator.main_process_first():
                self.tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
                self.text_encoder = CLIPTextModel.from_pretrained(
                    model_id, subfolder="text_encoder", torch_dtype=torch_dtype
                )
                self.vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae", torch_dtype=torch_dtype)
                # Keep trainable UNet in fp32; Accelerate autocast handles mixed precision safely.
                self.unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet")
                self.noise_scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")

            # Move frozen models to device
            self.vae.to(self.device)
            self.text_encoder.to(self.device)

            self.vae.requires_grad_(False)
            self.text_encoder.requires_grad_(False)
            self.unet.train()
            self._configure_trainable_parameters()

            trainable_params, total_params = self._count_trainable_params()

            # Memory optimizations (best-effort).
            try:
                self.unet.enable_gradient_checkpointing()
            except Exception as e:
                if self.accelerator.is_main_process:
                    self.logger.warning(f"UNet gradient checkpointing not enabled: {e}")

            if hasattr(self.unet, "enable_xformers_memory_efficient_attention"):
                try:
                    self.unet.enable_xformers_memory_efficient_attention()
                except Exception as e:
                    if self.accelerator.is_main_process:
                        self.logger.warning(f"xFormers memory efficient attention not enabled: {e}")
            elif hasattr(self.unet, "set_attention_slice"):
                try:
                    self.unet.set_attention_slice("auto")
                except Exception as e:
                    if self.accelerator.is_main_process:
                        self.logger.warning(f"UNet attention slicing not enabled: {e}")

            if hasattr(self.vae, "enable_slicing"):
                try:
                    self.vae.enable_slicing()
                except Exception:
                    pass

            if self.accelerator.is_main_process:
                self.logger.info(
                    f"Successfully loaded pretrained models from '{model_id}' (mixed_precision={mixed_precision})"
                )
                self.logger.info(
                    f"UNet trainable parameters: {trainable_params:,}/{total_params:,} "
                    f"(mode={self.conditioning_finetune_mode})"
                )
        except Exception as e:
            self.logger.error(f"Failed to load models. Error: {e}")
            raise

    @staticmethod
    def _sanitize_job_name(name: str) -> str:
        return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name)).strip("_") or "stage12_job"

    @staticmethod
    def _normalize_caption_text(value: str) -> str:
        value = str(value or "").replace("\n", " ")
        value = re.sub(r"\s+", " ", value)
        return value.strip()

    def _prepare_captions_for_clip(self, captions):
        if isinstance(captions, list):
            values = [self._normalize_caption_text(c) for c in captions]
        else:
            values = [self._normalize_caption_text(captions)]

        capped_values: list[str] = []
        truncated_count = 0
        max_tokens = min(int(self.prompt_token_budget), int(self.tokenizer.model_max_length))
        for value in values:
            capped, was_truncated = cap_prompt_to_token_budget(
                value,
                tokenizer=self.tokenizer,
                max_tokens=max_tokens,
            )
            if was_truncated:
                truncated_count += 1
            capped_values.append(self._normalize_caption_text(capped))

        self._prompt_seen_total += len(capped_values)
        self._prompt_truncated_total += truncated_count
        self._prompt_seen_epoch += len(capped_values)
        self._prompt_truncated_epoch += truncated_count
        return capped_values if isinstance(captions, list) else capped_values[0]

    def _configure_trainable_parameters(self):
        mode = self.conditioning_finetune_mode
        if mode in ("full", "full_unet"):
            self.unet.requires_grad_(True)
            return

        self.unet.requires_grad_(False)
        if mode in ("cross_attention", "cross_attention_only"):
            for name, param in self.unet.named_parameters():
                if "attn2" in name:
                    param.requires_grad = True
            return

        if mode in ("cross_attention_plus_output", "cross_attention_out"):
            for name, param in self.unet.named_parameters():
                if ("attn2" in name) or ("conv_out" in name) or ("conv_norm_out" in name):
                    param.requires_grad = True
            return

        if self.accelerator.is_main_process:
            self.logger.warning(
                f"Unknown conditioning_finetune_mode='{mode}'. Falling back to full_unet trainability."
            )
        self.unet.requires_grad_(True)

    def _count_trainable_params(self) -> tuple[int, int]:
        total = 0
        trainable = 0
        for param in self.unet.parameters():
            count = param.numel()
            total += count
            if param.requires_grad:
                trainable += count
        return trainable, total

    def _iter_trainable_named_parameters(self):
        model = self.accelerator.unwrap_model(self.unet)
        for name, param in model.named_parameters():
            if param.requires_grad:
                yield name, param

    def _init_ema_state(self):
        self._ema_state = {}
        if not self.ema_enabled:
            return
        for name, param in self._iter_trainable_named_parameters():
            self._ema_state[name] = param.detach().float().clone()

    def _load_ema_state(self, ema_state: dict | None):
        if not self.ema_enabled:
            self._ema_state = {}
            return
        self._init_ema_state()
        if not isinstance(ema_state, dict):
            return
        for name, value in ema_state.items():
            if name in self._ema_state and isinstance(value, torch.Tensor):
                self._ema_state[name] = value.detach().float().clone()

    def _update_ema_state(self):
        if not self.ema_enabled or (not self._ema_state):
            return
        self._optimizer_step += 1
        if self._optimizer_step <= self.ema_update_after_step:
            return
        decay = float(self.ema_decay)
        with torch.no_grad():
            for name, param in self._iter_trainable_named_parameters():
                if name not in self._ema_state:
                    self._ema_state[name] = param.detach().float().clone()
                    continue
                ema_value = self._ema_state[name]
                ema_value.mul_(decay).add_(param.detach().float(), alpha=(1.0 - decay))

    def _apply_ema_to_model(self) -> dict[str, torch.Tensor]:
        if not self.ema_enabled or (not self._ema_state):
            return {}
        backup: dict[str, torch.Tensor] = {}
        with torch.no_grad():
            for name, param in self._iter_trainable_named_parameters():
                if name not in self._ema_state:
                    continue
                backup[name] = param.detach().clone()
                param.copy_(self._ema_state[name].to(device=param.device, dtype=param.dtype))
        return backup

    def _restore_model_from_backup(self, backup: dict[str, torch.Tensor]):
        if not backup:
            return
        with torch.no_grad():
            for name, param in self._iter_trainable_named_parameters():
                if name in backup:
                    param.copy_(backup[name].to(device=param.device, dtype=param.dtype))

    def _log_prompt_truncation_stats(self, *, epoch_number: int):
        if (not self.accelerator.is_main_process) or (not self.log_prompt_truncation):
            return
        if self._prompt_seen_epoch <= 0:
            return
        rate = self._prompt_truncated_epoch / max(1, self._prompt_seen_epoch)
        self.logger.info(
            f"Epoch {epoch_number} | Prompt truncation: {self._prompt_truncated_epoch}/{self._prompt_seen_epoch} "
            f"({rate:.2%})"
        )

    def _load_epoch_checkpoint(self):
        if not self.resume_enabled or (not self.checkpoint_file.exists()):
            return None
        try:
            state = torch.load(self.checkpoint_file, map_location="cpu")
            if not isinstance(state, dict):
                return None
            return state
        except Exception as exc:
            if self.accelerator.is_main_process:
                self.logger.warning(f"Could not load checkpoint {self.checkpoint_file}: {exc}")
            return None

    def _save_epoch_checkpoint(
        self,
        *,
        epoch: int,
        best_val_loss: float,
        best_epoch: int,
        patience_counter: int,
        training_logs: list[dict],
        optimizer,
        lr_scheduler,
    ):
        if not self.accelerator.is_main_process:
            return
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "epoch": int(epoch),
            "best_val_loss": float(best_val_loss),
            "best_epoch": int(best_epoch),
            "patience_counter": int(patience_counter),
            "training_logs": list(training_logs),
            "model_state": self.accelerator.unwrap_model(self.unet).state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": lr_scheduler.state_dict(),
            "prompt_seen_total": int(self._prompt_seen_total),
            "prompt_truncated_total": int(self._prompt_truncated_total),
            "optimizer_step": int(self._optimizer_step),
        }
        if self.ema_enabled and self._ema_state:
            payload["ema_state"] = {k: v.detach().cpu() for k, v in self._ema_state.items()}
        scaler = getattr(self.accelerator, "scaler", None)
        if scaler is not None:
            try:
                payload["scaler_state"] = scaler.state_dict()
            except Exception:
                pass
        try:
            torch.save(payload, self.checkpoint_file)
        except Exception as exc:
            self.logger.warning(f"Could not save epoch checkpoint {self.checkpoint_file}: {exc}")

    def _clear_checkpoint(self):
        if not self.accelerator.is_main_process:
            return
        try:
            if self.checkpoint_file.exists():
                self.checkpoint_file.unlink()
            if self.checkpoint_dir.exists() and not any(self.checkpoint_dir.iterdir()):
                self.checkpoint_dir.rmdir()
        except Exception as exc:
            self.logger.warning(f"Could not clear checkpoint file(s): {exc}")

    def _compute_loss(self, batch):
        """Shared loss computation."""
        with torch.no_grad(), self.accelerator.autocast():
            latents = self.vae.encode(batch["original_image"]).latent_dist.sample() * self.vae.config.scaling_factor
            masked_latents = self.vae.encode(batch["masked_image"]).latent_dist.sample() * self.vae.config.scaling_factor

        # Text Encoding (Frozen)
        captions = self._prepare_captions_for_clip(batch["caption"])
        text_inputs = self.tokenizer(
            captions,
            padding="max_length",
            max_length=min(self.prompt_token_budget, self.tokenizer.model_max_length),
            truncation=True,
            return_tensors="pt",
        )
        with torch.no_grad(), self.accelerator.autocast():
            encoder_hidden_states = self.text_encoder(text_inputs.input_ids.to(self.device))[0]

        # Noise
        mask = F.interpolate(batch["mask"], size=latents.shape[-2:])
        mask = mask.to(dtype=latents.dtype)
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (bsz,),
            device=latents.device,
        ).long()
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

        # Predict
        latent_model_input = torch.cat([noisy_latents, mask, masked_latents], dim=1)
        with self.accelerator.autocast():
            noise_pred = self.unet(latent_model_input, timesteps, encoder_hidden_states).sample

        # Loss
        noise_pred = noise_pred.float()
        noise = noise.float()
        per_pixel_mse = (noise_pred - noise).pow(2)
        if self.mask_aware_loss:
            loss_mask = (mask > self.mask_loss_threshold).to(dtype=per_pixel_mse.dtype)
            loss_mask = loss_mask.expand_as(per_pixel_mse)
            inv_mask = 1.0 - loss_mask

            inside_denom = loss_mask.sum().clamp_min(1.0)
            outside_denom = inv_mask.sum().clamp_min(1.0)
            inside_loss = (per_pixel_mse * loss_mask).sum() / inside_denom
            outside_loss = (per_pixel_mse * inv_mask).sum() / outside_denom
            loss = (self.mask_loss_inside_weight * inside_loss) + (self.mask_loss_outside_weight * outside_loss)
        else:
            loss = per_pixel_mse.mean()
        return loss

    def train(self, train_dataloader, val_dataloader):
        self._load_pretrained_models()

        # --- Harden hyperparameter access: always use .get() with safe defaults ---
        lr = self.hyperparams.get("learning_rate", 1e-5)
        weight_decay = self.hyperparams.get("adam_weight_decay", 1e-2)

        if self.accelerator.is_main_process:
            self.logger.info(
                f"Hyperparameters: lr={lr}, weight_decay={weight_decay}, "
                f"scheduler={self.hyperparams.get('lr_scheduler', 'constant')}, "
                f"warmup={self.hyperparams.get('lr_warmup_steps', 0)}, "
                f"epochs={self.hyperparams.get('num_train_epochs', 10)}, "
                f"batch={self.hyperparams.get('train_batch_size', 4)}, "
                f"grad_clip={self.hyperparams.get('max_grad_norm', 1.0)}, "
                f"patience={self.hyperparams.get('early_stopping_patience', 5)}"
            )

        trainable_params = [p for p in self.unet.parameters() if p.requires_grad]
        if not trainable_params:
            raise RuntimeError(
                "No trainable UNet parameters remain after conditioning_finetune_mode filtering."
            )
        optimizer = AdamW(
            trainable_params,
            lr=lr,
            weight_decay=weight_decay,
        )

        hp_scheduler = self.hyperparams.get("lr_scheduler", "constant")
        hp_warmup = self.hyperparams.get("lr_warmup_steps", 0)
        num_epochs = self.hyperparams.get("num_train_epochs", self.config.num_epochs)
        total_steps = len(train_dataloader) * num_epochs

        lr_scheduler = get_scheduler(
            hp_scheduler,
            optimizer=optimizer,
            num_warmup_steps=hp_warmup,
            num_training_steps=total_steps,
        )

        # --- Gradient clipping threshold ---
        max_grad_norm = self.hyperparams.get("max_grad_norm", 1.0)

        # --- Early stopping ---
        patience = self.hyperparams.get("early_stopping_patience", 5)
        patience_counter = 0

        best_val_loss = float("inf")
        best_epoch = 0
        training_logs = []
        start_epoch = 0
        loaded_ema_state = None

        checkpoint_state = self._load_epoch_checkpoint()
        if checkpoint_state is not None:
            try:
                model_state = checkpoint_state.get("model_state")
                if model_state:
                    self.unet.load_state_dict(model_state, strict=False)
                if checkpoint_state.get("optimizer_state"):
                    optimizer.load_state_dict(checkpoint_state["optimizer_state"])
                if checkpoint_state.get("scheduler_state"):
                    lr_scheduler.load_state_dict(checkpoint_state["scheduler_state"])
                scaler_state = checkpoint_state.get("scaler_state")
                scaler = getattr(self.accelerator, "scaler", None)
                if scaler_state and scaler is not None:
                    try:
                        scaler.load_state_dict(scaler_state)
                    except Exception:
                        pass
                start_epoch = int(checkpoint_state.get("epoch", -1)) + 1
                best_val_loss = float(checkpoint_state.get("best_val_loss", float("inf")))
                best_epoch = int(checkpoint_state.get("best_epoch", 0))
                patience_counter = int(checkpoint_state.get("patience_counter", 0))
                restored_logs = checkpoint_state.get("training_logs", [])
                if isinstance(restored_logs, list):
                    training_logs = restored_logs
                self._prompt_seen_total = int(checkpoint_state.get("prompt_seen_total", 0))
                self._prompt_truncated_total = int(checkpoint_state.get("prompt_truncated_total", 0))
                self._optimizer_step = int(checkpoint_state.get("optimizer_step", 0))
                loaded_ema_state = checkpoint_state.get("ema_state")
                if self.accelerator.is_main_process:
                    self.logger.info(
                        f"Resuming epoch-level checkpoint at epoch {start_epoch + 1} "
                        f"for job '{self._sanitize_job_name(self.job_name)}'."
                    )
            except Exception as exc:
                if self.accelerator.is_main_process:
                    self.logger.warning(f"Checkpoint resume skipped due to state-load error: {exc}")
                start_epoch = 0
                best_val_loss = float("inf")
                best_epoch = 0
                patience_counter = 0
                training_logs = []
                loaded_ema_state = None
                self._prompt_seen_total = 0
                self._prompt_truncated_total = 0
                self._optimizer_step = 0

        # Prepare with Accelerate
        self.unet, optimizer, train_dataloader, val_dataloader, lr_scheduler = self.accelerator.prepare(
            self.unet,
            optimizer,
            train_dataloader,
            val_dataloader,
            lr_scheduler,
        )
        trainable_params = [p for p in self.unet.parameters() if p.requires_grad]
        self._load_ema_state(loaded_ema_state)

        last_train_loss = float("nan")
        last_val_loss = float("nan")
        epochs_ran = 0

        if self.accelerator.is_main_process:
            self.logger.info(
                f"Starting distributed training loop — "
                f"{num_epochs} epochs, scheduler={hp_scheduler}, "
                f"max_grad_norm={max_grad_norm}, patience={patience}, start_epoch={start_epoch + 1}"
            )

        for epoch in range(start_epoch, num_epochs):
            epochs_ran = epoch + 1
            self._prompt_seen_epoch = 0
            self._prompt_truncated_epoch = 0
            self.unet.train()
            epoch_loss = 0.0
            steps = 0

            progress_bar = tqdm(
                total=len(train_dataloader),
                desc=f"Epoch {epoch+1}",
                disable=not self.accelerator.is_main_process,
            )

            for batch in train_dataloader:
                with self.accelerator.accumulate(self.unet):
                    loss = self._compute_loss(batch)
                    self.accelerator.backward(loss)
                    if self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(trainable_params, max_grad_norm)
                    optimizer.step()
                    if self.accelerator.sync_gradients:
                        self._update_ema_state()
                    lr_scheduler.step()
                    optimizer.zero_grad(set_to_none=True)

                    epoch_loss += loss.item()
                    steps += 1

                progress_bar.update(1)
                progress_bar.set_postfix(loss=loss.item())

            progress_bar.close()

            # Validation
            avg_train_loss = epoch_loss / steps if steps > 0 else 0

            self.unet.eval()
            val_loss_accum = 0.0
            val_steps = 0
            ema_backup = {}
            if self.use_ema_for_validation:
                ema_backup = self._apply_ema_to_model()

            with torch.no_grad():
                for batch in val_dataloader:
                    loss = self._compute_loss(batch)
                    gathered_loss = self.accelerator.gather(loss)
                    val_loss_accum += gathered_loss.mean().item()
                    val_steps += 1
            if ema_backup:
                self._restore_model_from_backup(ema_backup)

            avg_val_loss = val_loss_accum / val_steps if val_steps > 0 else 0
            last_train_loss = float(avg_train_loss)
            last_val_loss = float(avg_val_loss)

            if self.accelerator.is_main_process:
                self.logger.info(f"Epoch {epoch + 1} | Train Loss: {avg_train_loss:.5f} | Val Loss: {avg_val_loss:.5f}")
                self._log_prompt_truncation_stats(epoch_number=epoch + 1)
                prompt_trunc_rate = self._prompt_truncated_epoch / max(1, self._prompt_seen_epoch)
                training_logs.append(
                    {
                        "epoch": epoch + 1,
                        "train_loss": avg_train_loss,
                        "val_loss": avg_val_loss,
                        "prompt_truncation_rate": prompt_trunc_rate,
                    }
                )

            # IMPORTANT: early-stopping state must be updated on ALL ranks.
            improved = avg_val_loss < best_val_loss
            if improved:
                best_val_loss = avg_val_loss
                best_epoch = epoch + 1
                patience_counter = 0
                if self.accelerator.is_main_process:
                    save_backup = {}
                    if self.use_ema_for_saving:
                        save_backup = self._apply_ema_to_model()
                    save_path = self.output_dir / "unet_best"
                    self.accelerator.unwrap_model(self.unet).save_pretrained(save_path)
                    if save_backup:
                        self._restore_model_from_backup(save_backup)
                    self.logger.info(f"Saved best model to {save_path}")
            else:
                patience_counter += 1
                if self.accelerator.is_main_process:
                    self.logger.info(f"No improvement for {patience_counter}/{patience} epochs.")

            if (epoch + 1) % self.checkpoint_every_epochs == 0:
                self._save_epoch_checkpoint(
                    epoch=epoch,
                    best_val_loss=best_val_loss,
                    best_epoch=best_epoch,
                    patience_counter=patience_counter,
                    training_logs=training_logs,
                    optimizer=optimizer,
                    lr_scheduler=lr_scheduler,
                )
            self.accelerator.wait_for_everyone()

            if patience_counter >= patience:
                if self.accelerator.is_main_process:
                    self.logger.info(
                        f"Early stopping triggered after {epoch + 1} epochs "
                        f"(patience={patience})."
                    )
                self.accelerator.wait_for_everyone()
                break

        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            # Save Final Model (Always)
            final_backup = {}
            if self.use_ema_for_saving:
                final_backup = self._apply_ema_to_model()
            final_path = self.output_dir / "unet_final"
            self.accelerator.unwrap_model(self.unet).save_pretrained(final_path)
            if final_backup:
                self._restore_model_from_backup(final_backup)
            self.logger.info(f"Saved final model to {final_path}")

            if training_logs:
                df = pd.DataFrame(training_logs)
                if self.plotter:
                    df.to_csv(self.plotter.output_dir / "training_logs.csv", index=False)
                    self._plot_loss_curves(df, self.plotter.output_dir)
                else:
                    df.to_csv(self.output_dir / "training_logs.csv", index=False)
                    self._plot_loss_curves(df, self.output_dir)
            self._clear_checkpoint()

        return {
            "best_val_loss": float(best_val_loss),
            "best_epoch": int(best_epoch),
            "epochs_ran": int(epochs_ran),
            "final_train_loss": float(last_train_loss),
            "final_val_loss": float(last_val_loss),
            "prompt_truncation_rate_total": (
                float(self._prompt_truncated_total / max(1, self._prompt_seen_total))
            ),
        }

    def _plot_loss_curves(self, df, output_dir):
        """Generate training and validation loss curve plot."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(df["epoch"], df["train_loss"], "b-o", label="Train Loss", markersize=4)
        ax.plot(df["epoch"], df["val_loss"], "r-s", label="Val Loss", markersize=4)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss (MSE)")
        ax.set_title("Training & Validation Loss Curves")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Annotate best epoch
        best_idx = df["val_loss"].idxmin()
        best_epoch = df.loc[best_idx, "epoch"]
        best_val = df.loc[best_idx, "val_loss"]
        ax.axvline(x=best_epoch, color="green", linestyle="--", alpha=0.5, label=f"Best epoch={best_epoch}")
        ax.annotate(
            f"Best: {best_val:.5f}",
            xy=(best_epoch, best_val),
            xytext=(best_epoch + 0.5, best_val + 0.002),
            arrowprops=dict(arrowstyle="->", color="green"),
            fontsize=9,
            color="green",
        )

        plt.tight_layout()
        plt.savefig(output_dir / "loss_curves.png", dpi=150)
        plt.close()
