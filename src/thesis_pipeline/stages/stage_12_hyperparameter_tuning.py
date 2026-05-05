"""Stage S12 - Hyperparameter plan generation."""

import hashlib
import itertools
import json
import logging
from pathlib import Path

import pandas as pd

from thesis_pipeline.config_manager import ConfigManager
from thesis_pipeline.utils.common import save_yaml
from thesis_pipeline.utils.stage_artifacts import resolve_stage_artifact_dir
from thesis_pipeline.visualization import ThesisPlotter


class HyperparameterTuningStage:
    """Create a deterministic hyperparameter plan for Stage 12.

    In `mini_sweep` mode this stage writes:
    - `sweep_plan.yaml` with metadata and trials.
    - `sweep_plan.csv` for easy inspection.
    - `best_hyperparameters.yaml` as a compatibility fallback placeholder.
    """

    def __init__(self, config_manager: ConfigManager):
        self.config_manager = config_manager
        self.config = config_manager.get_hyperparameter_tuning_config()
        self.paths = config_manager.get_paths()
        self.logger = logging.getLogger(__name__)
        self.output_dir = resolve_stage_artifact_dir(self.config_manager, "S12")

    @staticmethod
    def _clamp(value: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, float(value)))

    @staticmethod
    def _base_training_hparams() -> dict:
        return {
            "train_batch_size": 4,
            "num_train_epochs": 10,
            "gradient_accumulation_steps": 1,
            "lr_scheduler": "cosine",
            "lr_warmup_steps": 100,
            "max_grad_norm": 1.0,
        }

    def _build_mini_sweep_trials(self, lr_min: float, lr_max: float) -> list[dict]:
        mini_cfg = self.config.get("mini_sweep", {})
        lr_values = mini_cfg.get("lr_values", [1e-5])
        wd_values = mini_cfg.get("adam_weight_decay_values", [1e-2])
        warmup_values = mini_cfg.get("lr_warmup_steps_values", [100])
        grad_norm_values = mini_cfg.get("max_grad_norm_values", [1.0])
        total_candidates = len(lr_values) * len(wd_values) * len(warmup_values) * len(grad_norm_values)
        max_trials = int(self.config.get("max_trials", total_candidates))
        patience = int(mini_cfg.get("early_stopping_patience", 3))

        candidates: list[dict] = []
        for warmup, wd, grad_norm, lr in itertools.product(
            warmup_values, wd_values, grad_norm_values, lr_values
        ):
            candidates.append(
                {
                    **self._base_training_hparams(),
                    "learning_rate": self._clamp(float(lr), lr_min, lr_max),
                    "adam_weight_decay": float(wd),
                    "lr_warmup_steps": int(warmup),
                    "max_grad_norm": float(grad_norm),
                    "early_stopping_patience": patience,
                }
            )

        if not candidates:
            return []

        if max_trials >= len(candidates):
            selected = candidates
        elif max_trials <= 1:
            selected = [candidates[0]]
        else:
            index_positions = {
                int(round(i * (len(candidates) - 1) / (max_trials - 1))) for i in range(max_trials)
            }
            selected = [candidates[idx] for idx in sorted(index_positions)]

            # Guard for duplicate rounded positions in edge cases.
            cursor = 0
            while len(selected) < max_trials and cursor < len(candidates):
                candidate = candidates[cursor]
                if candidate not in selected:
                    selected.append(candidate)
                cursor += 1

        rows: list[dict] = []
        for trial_num, row in enumerate(selected[:max_trials], start=1):
            rows.append({"trial_id": f"trial_{trial_num:02d}", **row})
        return rows

    def _best_fallback_from_trials(self, trials: list[dict], lr_min: float, lr_max: float) -> dict:
        if not trials:
            best_lr = self._clamp(1e-5, lr_min, lr_max)
            return {
                "selection_method": "literature_informed",
                "learning_rate": best_lr,
                "adam_weight_decay": 1e-2,
                "early_stopping_patience": 5,
                **self._base_training_hparams(),
            }

        # Compatibility fallback: pick trial closest to canonical SD FT recipe.
        selected = min(
            trials,
            key=lambda t: abs(float(t["learning_rate"]) - 1e-5)
            + abs(float(t["adam_weight_decay"]) - 1e-2),
        )
        return {
            "selection_method": "mini_sweep_pending_stage12",
            "selected_trial": selected["trial_id"],
            "learning_rate": float(selected["learning_rate"]),
            "adam_weight_decay": float(selected["adam_weight_decay"]),
            "early_stopping_patience": int(selected["early_stopping_patience"]),
            **self._base_training_hparams(),
        }

    def run(self):
        self.logger.info("=" * 20 + " STAGE 12: Hyperparameter Planning " + "=" * 20)
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)

            lr_min = float(self.config.learning_rate_range[0])
            lr_max = float(self.config.learning_rate_range[1])
            mode = str(self.config.get("mode", "literature_informed")).strip().lower()

            trials: list[dict] = []
            if mode == "mini_sweep":
                trials = self._build_mini_sweep_trials(lr_min, lr_max)
                if not trials:
                    raise RuntimeError("mini_sweep mode produced zero trials.")

                sweep_payload = {
                    "selection_method": "mini_sweep",
                    "selection_metric": str(
                        self.config.get("mini_sweep", {}).get("selection_metric", "best_val_loss")
                    ),
                    "search_space": {
                        "learning_rate_range": [lr_min, lr_max],
                        "lr_values": list(self.config.get("mini_sweep", {}).get("lr_values", [])),
                        "adam_weight_decay_values": list(
                            self.config.get("mini_sweep", {}).get("adam_weight_decay_values", [])
                        ),
                        "lr_warmup_steps_values": list(
                            self.config.get("mini_sweep", {}).get("lr_warmup_steps_values", [100])
                        ),
                        "max_grad_norm_values": list(
                            self.config.get("mini_sweep", {}).get("max_grad_norm_values", [1.0])
                        ),
                        "max_trials": int(self.config.get("max_trials", len(trials))),
                    },
                    "trials": trials,
                }
                immutable_plan_blob = json.dumps(trials, sort_keys=True, separators=(",", ":"))
                sweep_payload["plan_sha256"] = hashlib.sha256(immutable_plan_blob.encode("utf-8")).hexdigest()
                save_yaml(self.output_dir / "sweep_plan.yaml", sweep_payload)
                pd.DataFrame(trials).to_csv(self.output_dir / "sweep_plan.csv", index=False)
                self.logger.info(f"Mini-sweep plan saved with {len(trials)} trials.")

            best_params = self._best_fallback_from_trials(trials, lr_min, lr_max)
            best_params["search_space"] = {
                "learning_rate_range": [lr_min, lr_max],
                "max_trials": int(self.config.get("max_trials", 5)),
            }
            best_params["references"] = [
                "Rombach et al. 2022 (LDM / SD training recipe)",
                "Ruiz et al. 2023 (DreamBooth fine-tuning at 1e-5)",
                "HuggingFace Diffusers v0.25 fine-tuning guide",
            ]
            if trials:
                immutable_plan_blob = json.dumps(trials, sort_keys=True, separators=(",", ":"))
                best_params["source_plan_sha256"] = hashlib.sha256(
                    immutable_plan_blob.encode("utf-8")
                ).hexdigest()

            output_file = self.output_dir / "best_hyperparameters.yaml"
            save_yaml(output_file, best_params)
            self.logger.info(f"Hyperparameter plan saved to {output_file}")

            plotter = ThesisPlotter(self.output_dir)
            display_params = {
                k: v
                for k, v in best_params.items()
                if k not in ("search_space", "references", "selection_method")
            }
            df = pd.DataFrame(list(display_params.items()), columns=["Parameter", "Value"])
            plotter.plot_table(df, "Selected Hyperparameters (Plan)", "hyperparameters_table")
            plotter.plot_hyperparameter_summary(best_params)
            plotter.plot_lr_schedule(
                total_epochs=int(best_params.get("num_train_epochs", 10)),
                lr_max=float(best_params["learning_rate"]),
            )

            self.logger.info("=" * 20 + " STAGE 12 COMPLETED " + "=" * 20 + "\\n")
        except Exception as exc:
            self.logger.exception(f"Error in HP planning stage: {exc}")
            raise


if __name__ == "__main__":
    cm = ConfigManager()
    HyperparameterTuningStage(cm).run()
