import json
import logging
import math
import os
import random
import shutil
import subprocess
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
from accelerate import Accelerator
from PIL import Image
from sklearn.model_selection import KFold
from torch.utils.data import ConcatDataset, DataLoader, Subset

from thesis_pipeline.components.dataset import InpaintingDataset
from thesis_pipeline.components.model_training import ModelTrainer
from thesis_pipeline.config_manager import ConfigManager
from thesis_pipeline.utils.common import load_yaml, save_yaml
from thesis_pipeline.visualization import ThesisPlotter


class ModelTrainingStage:
    def __init__(self, config_manager: ConfigManager):
        self.config_manager = config_manager
        self.config = config_manager.get_training_config()
        root_config = getattr(config_manager, "config", {})
        if hasattr(root_config, "get"):
            runtime_raw = root_config.get("model_training", {})
        elif isinstance(root_config, dict):
            runtime_raw = root_config.get("model_training", {})
        else:
            runtime_raw = {}
        self.runtime_cfg = self._to_plain_dict(runtime_raw)
        self.paths = config_manager.get_paths()
        self.dp_config = config_manager.get_data_processing_config()
        self.global_seed = int(getattr(config_manager.get_global_params(), "random_state", 42))
        self.logger = logging.getLogger(__name__)
        self.is_primary = self._is_primary_process()

    @staticmethod
    def _as_int_env(name: str, default: int) -> int:
        value = os.environ.get(name)
        try:
            return int(value) if value is not None else default
        except Exception:
            return default

    @classmethod
    def _is_primary_process(cls) -> bool:
        rank = cls._as_int_env("RANK", 0)
        local_rank = cls._as_int_env("LOCAL_RANK", 0)
        return rank == 0 and local_rank in (0, -1)

    @staticmethod
    def _defaults() -> dict:
        return {
            "learning_rate": 1e-5,
            "train_batch_size": 4,
            "num_train_epochs": 10,
            "gradient_accumulation_steps": 1,
            "lr_scheduler": "cosine",
            "lr_warmup_steps": 100,
            "adam_weight_decay": 1e-2,
            "max_grad_norm": 1.0,
            "early_stopping_patience": 5,
        }

    @staticmethod
    def _to_plain_dict(value) -> dict:
        if value is None:
            return {}
        if hasattr(value, "to_dict"):
            return value.to_dict()
        if isinstance(value, dict):
            return value
        return dict(value)

    @staticmethod
    def _is_finite_number(value) -> bool:
        try:
            return math.isfinite(float(value))
        except Exception:
            return False

    @classmethod
    def _select_best_trial(cls, trial_rows: list[dict]) -> dict | None:
        if not trial_rows:
            return None
        valid = [r for r in trial_rows if cls._is_finite_number(r.get("best_val_loss"))]
        pool = valid if valid else trial_rows
        return min(pool, key=lambda r: float(r.get("best_val_loss", float("inf"))))

    def _build_accelerator(self, hparams: dict) -> Accelerator:
        grad_accum = int(hparams.get("gradient_accumulation_steps", 1))
        return Accelerator(
            mixed_precision=self.config.get("mixed_precision", "fp16"),
            log_with="all",
            project_dir=self.paths.artifacts.logs,
            gradient_accumulation_steps=grad_accum,
            rng_types=[],
        )

    def _runtime_flag(self, key: str, default):
        value = self.runtime_cfg.get(key, default)
        if isinstance(default, bool):
            return bool(value)
        if isinstance(default, int):
            try:
                return int(value)
            except Exception:
                return default
        return value

    @staticmethod
    def _now_iso() -> str:
        return datetime.now().isoformat()

    def _maybe_log_gpu_snapshot(self, prefix: str):
        if not self.is_primary:
            return
        try:
            result = subprocess.run(
                ["nvidia-smi"],
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            if result.returncode == 0 and result.stdout:
                self.logger.info(f"{prefix} GPU snapshot:\n{result.stdout}")
            else:
                self.logger.info(f"{prefix} GPU snapshot unavailable (nvidia-smi rc={result.returncode}).")
        except Exception as exc:
            self.logger.info(f"{prefix} GPU snapshot failed: {exc}")

    def _run_training_job(
        self,
        hparams: dict,
        model_dir: Path,
        report_dir: Path,
        train_dataset,
        val_dataset,
        *,
        job_label: str,
    ) -> dict:
        accelerator = self._build_accelerator(hparams)

        if accelerator.is_main_process:
            self.logger.info(f"Stage S13 job '{job_label}': starting")

        model_dir.mkdir(parents=True, exist_ok=True)
        report_dir.mkdir(parents=True, exist_ok=True)

        trainer = ModelTrainer(
            self.config,
            hparams,
            model_dir,
            report_dir,
            accelerator,
            runtime_config=self.runtime_cfg,
            job_name=job_label,
        )
        batch_size = int(hparams.get("train_batch_size", self.config.train_batch_size))
        num_workers = int(self._runtime_flag("dataloader_num_workers", 4))
        pin_memory = bool(self._runtime_flag("dataloader_pin_memory", True))
        persistent_workers = bool(self._runtime_flag("dataloader_persistent_workers", num_workers > 0))

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=(persistent_workers and num_workers > 0),
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=(persistent_workers and num_workers > 0),
        )

        t0 = time.perf_counter()
        summary = trainer.train(
            train_loader,
            val_loader,
        )
        accelerator.wait_for_everyone()
        duration_seconds = time.perf_counter() - t0
        summary = dict(summary or {})
        summary["duration_seconds"] = float(duration_seconds)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if accelerator.is_main_process:
            self.logger.info(
                f"Stage S13 job '{job_label}': completed "
                f"(best_val_loss={summary.get('best_val_loss')}, best_epoch={summary.get('best_epoch')})"
            )

        return summary

    def _generate_sweep_visuals(self, report_root: Path, trial_rows: list[dict]):
        if not self.is_primary or not trial_rows:
            return
        plotter = ThesisPlotter(report_root / "charts")
        df = pd.DataFrame(trial_rows)
        if {"best_val_loss", "duration_seconds"}.issubset(df.columns):
            plotter.plot_sweep_pareto(df)
        if {"learning_rate", "adam_weight_decay", "best_val_loss"}.issubset(df.columns):
            plotter.plot_lr_wd_response_heatmap(df)
        plotter.plot_trial_learning_curves_panel(report_root / "sweep")

    def _generate_kfold_visuals(self, report_root: Path, fold_rows: list[dict]):
        if not self.is_primary or not fold_rows:
            return
        plotter = ThesisPlotter(report_root / "charts")
        plotter.plot_kfold_training_stability(pd.DataFrame(fold_rows))

    @staticmethod
    def _compute_pairwise_interactions(df: pd.DataFrame, *, target_col: str, factors: list[str]) -> list[dict]:
        rows: list[dict] = []
        if target_col not in df.columns:
            return rows

        work = df.copy()
        work[target_col] = pd.to_numeric(work[target_col], errors="coerce")
        work = work[work[target_col].notna()]
        if work.empty:
            return rows

        global_mean = float(work[target_col].mean())
        for i, a_col in enumerate(factors):
            for b_col in factors[i + 1 :]:
                if a_col not in work.columns or b_col not in work.columns:
                    continue
                if work[a_col].nunique(dropna=True) < 2 or work[b_col].nunique(dropna=True) < 2:
                    continue

                a_means = work.groupby(a_col)[target_col].mean().to_dict()
                b_means = work.groupby(b_col)[target_col].mean().to_dict()
                combo = work.groupby([a_col, b_col])[target_col].mean().reset_index(name="combo_mean")

                for _, row in combo.iterrows():
                    a_val = row[a_col]
                    b_val = row[b_col]
                    combo_mean = float(row["combo_mean"])
                    expected_additive = float(a_means[a_val] + b_means[b_val] - global_mean)
                    interaction = combo_mean - expected_additive
                    rows.append(
                        {
                            "factor_a": a_col,
                            "level_a": str(a_val),
                            "factor_b": b_col,
                            "level_b": str(b_val),
                            "combo_mean": combo_mean,
                            "expected_additive": expected_additive,
                            "interaction_effect": float(interaction),
                            "abs_interaction_effect": float(abs(interaction)),
                        }
                    )
        return rows

    def _write_conditioning_interaction_report(self, report_root: Path, trial_rows: list[dict]):
        if not self.is_primary or not trial_rows:
            return
        cfg = self.runtime_cfg.get("conditioning_interaction_analysis", {})
        if isinstance(cfg, bool):
            enabled = bool(cfg)
            max_gap = 0.05
        else:
            enabled = bool(self._to_plain_dict(cfg).get("enabled", False))
            max_gap = float(self._to_plain_dict(cfg).get("max_abs_interaction", 0.05))
        if not enabled:
            return

        df = pd.DataFrame(trial_rows)
        candidate_factors = [
            "conditioning_finetune_mode",
            "mask_aware_loss",
            "lr_scheduler",
            "adam_weight_decay",
            "max_grad_norm",
        ]
        factors = [c for c in candidate_factors if c in df.columns and df[c].nunique(dropna=True) >= 2]

        interactions = self._compute_pairwise_interactions(
            df,
            target_col="best_val_loss",
            factors=factors,
        )
        out_dir = report_root / "interaction_analysis"
        out_dir.mkdir(parents=True, exist_ok=True)

        summary = {
            "enabled": True,
            "target_metric": "best_val_loss",
            "factors_used": factors,
            "max_abs_interaction_threshold": float(max_gap),
            "pairwise_rows": len(interactions),
        }
        if interactions:
            max_abs = max(float(r["abs_interaction_effect"]) for r in interactions)
            summary["max_abs_interaction_observed"] = float(max_abs)
            summary["threshold_exceeded"] = bool(max_abs > max_gap)
            pd.DataFrame(interactions).to_csv(out_dir / "pairwise_interactions.csv", index=False)
        else:
            summary["max_abs_interaction_observed"] = None
            summary["threshold_exceeded"] = False

        (out_dir / "conditioning_interaction_summary.json").write_text(
            json.dumps(summary, indent=2),
            encoding="utf-8",
        )

        if bool(summary.get("threshold_exceeded", False)) and bool(
            self.config_manager.config.get("pipeline", {}).get("strict_fail_policy", False)
        ):
            raise RuntimeError(
                "Conditioning interaction analysis exceeded configured threshold: "
                f"{summary['max_abs_interaction_observed']:.6f} > {max_gap:.6f}"
            )

    def _run_regime_comparison_experiment(
        self,
        *,
        models_root: Path,
        report_root: Path,
        hparams: dict,
        base_train_dataset,
        val_dataset,
    ) -> dict[str, object] | None:
        cfg = self._to_plain_dict(self.runtime_cfg.get("regime_comparison", {}))
        enabled = bool(cfg.get("enabled", False))
        if not enabled:
            return None

        strict_fail = bool(cfg.get("strict_fail", True)) and bool(
            self.config_manager.config.get("pipeline", {}).get("strict_fail_policy", False)
        )
        compare_metric = str(cfg.get("compare_metric", "best_val_loss") or "best_val_loss")
        biased_overrides = self._to_plain_dict(cfg.get("biased_regime_overrides", {}))
        balanced_overrides = self._to_plain_dict(cfg.get("balanced_regime_overrides", {}))
        source_cfg = self._to_plain_dict(cfg.get("source_sampling", {}))

        regimes = [
            ("biased_regime", "biased", biased_overrides),
            ("balanced_regime", "balanced", balanced_overrides),
        ]
        original_runtime = dict(self.runtime_cfg)
        results: dict[str, dict] = {}

        try:
            for regime_name, source_mode, extra_overrides in regimes:
                self.runtime_cfg.update(original_runtime)
                for key, value in extra_overrides.items():
                    self.runtime_cfg[key] = value

                regime_model_dir = models_root / "regimes" / regime_name
                regime_report_dir = report_root / "regimes" / regime_name
                train_dataset, source_provenance = self._build_source_regime_subset(
                    base_train_dataset,
                    regime_mode=source_mode,
                    report_root=regime_report_dir,
                    source_cfg=source_cfg,
                )

                if bool(self.runtime_cfg.get("balanced_mask_sampling", False)):
                    train_dataset = self._build_balanced_train_subset(train_dataset, regime_report_dir)

                summary = self._run_training_job(
                    hparams,
                    regime_model_dir,
                    regime_report_dir,
                    train_dataset,
                    val_dataset,
                    job_label=f"regime:{regime_name}",
                )
                results[regime_name] = {
                    **summary,
                    "model_dir": str(regime_model_dir),
                    "report_dir": str(regime_report_dir),
                    "balanced_mask_sampling": bool(self.runtime_cfg.get("balanced_mask_sampling", False)),
                    "train_size": int(len(train_dataset)),
                    "source_sampling": source_provenance,
                }
        finally:
            self.runtime_cfg = original_runtime

        biased = results.get("biased_regime", {})
        balanced = results.get("balanced_regime", {})
        biased_metric = float(biased.get(compare_metric, float("nan")))
        balanced_metric = float(balanced.get(compare_metric, float("nan")))
        metric_delta = balanced_metric - biased_metric if math.isfinite(biased_metric) and math.isfinite(balanced_metric) else None

        regime_report = {
            "enabled": True,
            "compare_metric": compare_metric,
            "biased_regime": biased,
            "balanced_regime": balanced,
            "metric_delta_balanced_minus_biased": metric_delta,
            "distinct_checkpoint_roots": bool(biased.get("model_dir") and balanced.get("model_dir") and biased.get("model_dir") != balanced.get("model_dir")),
            "strict_fail": strict_fail,
        }

        out_dir = report_root / "interaction_analysis"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "regime_comparison_report.json").write_text(
            json.dumps(regime_report, indent=2),
            encoding="utf-8",
        )

        rows = []
        for regime_name in ("biased_regime", "balanced_regime"):
            payload = dict(results.get(regime_name, {}))
            payload["regime"] = regime_name
            rows.append(payload)
        pd.DataFrame(rows).to_csv(out_dir / "regime_comparison_summary.csv", index=False)

        if strict_fail:
            missing = [name for name in ("biased_regime", "balanced_regime") if name not in results]
            if missing:
                raise RuntimeError(f"Regime comparison missing results for {missing}")

        return regime_report

    @staticmethod
    def _infer_source_from_stem(stem: str) -> str:
        text = str(stem or "").lower()
        if text.startswith("wiki_"):
            return "wikimedia"
        if text.startswith("eur_"):
            return "europeana"
        return "unknown"

    def _build_source_regime_subset(
        self,
        train_dataset,
        *,
        regime_mode: str,
        report_root: Path,
        source_cfg: dict,
    ):
        if not hasattr(train_dataset, "samples"):
            provenance = {
                "regime_mode": regime_mode,
                "available": False,
                "reason": "dataset_has_no_samples_attribute",
                "selected_count": int(len(train_dataset)) if hasattr(train_dataset, "__len__") else None,
            }
            return train_dataset, provenance

        samples = list(getattr(train_dataset, "samples", []))
        if not samples:
            provenance = {
                "regime_mode": regime_mode,
                "available": False,
                "reason": "no_samples",
                "selected_count": 0,
            }
            return train_dataset, provenance

        ratio = int(source_cfg.get("wikimedia_to_europeana_ratio", 261) or 261)
        seed = int(source_cfg.get("seed", self.global_seed) or self.global_seed)
        rng = random.Random(seed ^ (0xB1A5 if regime_mode == "biased" else 0xBA1A))

        wiki_idx: list[int] = []
        eur_idx: list[int] = []
        unknown_idx: list[int] = []
        for idx, sample in enumerate(samples):
            image_path = sample[0]
            source = self._infer_source_from_stem(Path(image_path).stem)
            if source == "wikimedia":
                wiki_idx.append(idx)
            elif source == "europeana":
                eur_idx.append(idx)
            else:
                unknown_idx.append(idx)

        def _pick(indices: list[int], k: int) -> list[int]:
            if k <= 0:
                return []
            if len(indices) <= k:
                return list(indices)
            return rng.sample(indices, k=k)

        selected: list[int] = []
        if regime_mode == "balanced":
            n = min(len(wiki_idx), len(eur_idx))
            selected = _pick(wiki_idx, n) + _pick(eur_idx, n)
            selected.extend(unknown_idx)
        else:
            # Reconstruct V7-like source imbalance using a composition-controlled subset.
            if wiki_idx and eur_idx:
                eur_target = max(1, min(len(eur_idx), len(wiki_idx) // max(1, ratio)))
                wiki_target = min(len(wiki_idx), eur_target * max(1, ratio))
                selected = _pick(wiki_idx, wiki_target) + _pick(eur_idx, eur_target)
            else:
                selected = list(wiki_idx) + list(eur_idx)
            selected.extend(unknown_idx)

        rng.shuffle(selected)
        subset = Subset(train_dataset, selected)

        selected_sources = {"wikimedia": 0, "europeana": 0, "unknown": 0}
        for idx in selected:
            source = self._infer_source_from_stem(Path(samples[idx][0]).stem)
            selected_sources[source] = int(selected_sources.get(source, 0) + 1)

        wiki_count = int(selected_sources.get("wikimedia", 0))
        eur_count = int(selected_sources.get("europeana", 0))
        actual_ratio = (float(wiki_count) / float(eur_count)) if eur_count > 0 else None

        provenance = {
            "regime_mode": regime_mode,
            "available": True,
            "target_wikimedia_to_europeana_ratio": int(ratio),
            "source_counts_input": {
                "wikimedia": int(len(wiki_idx)),
                "europeana": int(len(eur_idx)),
                "unknown": int(len(unknown_idx)),
            },
            "source_counts_selected": selected_sources,
            "actual_wikimedia_to_europeana_ratio": actual_ratio,
            "selected_count": int(len(selected)),
        }

        report_root.mkdir(parents=True, exist_ok=True)
        (report_root / "source_sampling_provenance.json").write_text(
            json.dumps(provenance, indent=2),
            encoding="utf-8",
        )
        return subset, provenance

    def _load_fallback_hyperparams(self) -> dict:
        hyperparams_path = self.config_manager.get_stage_artifact_path("S12", "best_hyperparameters.yaml")
        if hyperparams_path.exists():
            loaded = self._to_plain_dict(load_yaml(hyperparams_path))
            self.logger.info(f"Loaded hyperparameters from {hyperparams_path}")
            return loaded

        self.logger.warning(f"Hyperparameters file not found at {hyperparams_path}. Using defaults.")
        return self._defaults()

    def _load_sweep_trials(self) -> list[dict]:
        sweep_path = self.config_manager.get_stage_artifact_path("S12", "sweep_plan.yaml")
        if not sweep_path.exists():
            return []

        payload = self._to_plain_dict(load_yaml(sweep_path))
        trials = payload.get("trials", [])
        if not isinstance(trials, list):
            return []

        normalized = []
        for idx, row in enumerate(trials, start=1):
            item = self._to_plain_dict(row)
            item.setdefault("trial_id", f"trial_{idx:02d}")
            normalized.append(item)
        return normalized

    @staticmethod
    def _copy_model_dir(src: Path, dst: Path):
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)

    @staticmethod
    def _infer_mask_type(stem: str) -> str:
        text = stem.lower()
        if "irregular" in text:
            return "irregular"
        if "edge" in text:
            return "edge"
        if ("rect" in text) or ("box" in text):
            return "rect"
        return "unknown"

    @staticmethod
    def _mask_coverage_ratio(mask_path: Path) -> float:
        try:
            with Image.open(mask_path) as image:
                hist = image.convert("L").histogram()
            total = float(sum(hist))
            if total <= 0:
                return 0.0
            masked = float(sum(hist[128:]))
            return masked / total
        except Exception:
            return 0.0

    @staticmethod
    def _coverage_bucket(ratio: float, thresholds: tuple[float, float]) -> str:
        low_cut, high_cut = thresholds
        if ratio < low_cut:
            return "low"
        if ratio < high_cut:
            return "medium"
        return "high"

    def _build_balanced_train_subset(self, train_dataset: InpaintingDataset, report_root: Path):
        if not hasattr(train_dataset, "samples"):
            return train_dataset
        samples = list(getattr(train_dataset, "samples", []))
        if not samples:
            return train_dataset

        thresholds_cfg = self.runtime_cfg.get("balanced_sampling_area_thresholds", [0.12, 0.30])
        try:
            low_cut = float(thresholds_cfg[0])
            high_cut = float(thresholds_cfg[1])
        except Exception:
            low_cut, high_cut = 0.12, 0.30
        thresholds = (low_cut, high_cut)

        cache_path = report_root / "mask_sampling_cache_train.json"
        cache = {}
        if cache_path.exists():
            try:
                loaded = json.loads(cache_path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    cache = loaded
            except Exception:
                cache = {}

        groups: dict[str, list[int]] = defaultdict(list)
        cache_updated = False
        for idx, sample in enumerate(samples):
            image_path, mask_path, _caption_path = sample
            stem = image_path.stem
            entry = cache.get(stem)
            if not isinstance(entry, dict):
                entry = {}

            mask_type = str(entry.get("mask_type") or self._infer_mask_type(stem))
            if "coverage_ratio" in entry:
                try:
                    coverage_ratio = float(entry["coverage_ratio"])
                except Exception:
                    coverage_ratio = self._mask_coverage_ratio(mask_path)
                    cache_updated = True
            else:
                coverage_ratio = self._mask_coverage_ratio(mask_path)
                cache_updated = True

            severity = self._coverage_bucket(coverage_ratio, thresholds)
            cache[stem] = {
                "mask_type": mask_type,
                "coverage_ratio": coverage_ratio,
                "severity": severity,
            }
            groups[f"{mask_type}|{severity}"].append(idx)

        if cache_updated and self.is_primary:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(cache, indent=2), encoding="utf-8")

        if len(groups) <= 1:
            self.logger.info("Balanced mask sampling skipped: only one mask bucket present in train split.")
            return train_dataset

        target_mode = str(self.runtime_cfg.get("balanced_sampling_target", "median")).strip().lower()
        counts = sorted(len(v) for v in groups.values())
        if target_mode == "max":
            target_count = counts[-1]
        elif target_mode == "mean":
            target_count = int(round(sum(counts) / max(1, len(counts))))
        else:
            target_count = counts[len(counts) // 2]
        target_count = max(1, target_count)

        seed = int(self.runtime_cfg.get("balanced_sampling_seed", self.global_seed))
        rng = random.Random(seed)
        balanced_indices: list[int] = []
        bucket_summary: dict[str, dict] = {}

        for bucket_key, indices in sorted(groups.items()):
            bucket_summary[bucket_key] = {"original": len(indices), "target": target_count}
            if len(indices) == target_count:
                selected = list(indices)
            elif len(indices) > target_count:
                selected = rng.sample(indices, k=target_count)
            else:
                selected = list(indices)
                selected.extend(rng.choices(indices, k=(target_count - len(indices))))
            balanced_indices.extend(selected)
            bucket_summary[bucket_key]["selected"] = len(selected)

        rng.shuffle(balanced_indices)
        subset = Subset(train_dataset, balanced_indices)
        if self.is_primary:
            self.logger.info(
                f"Balanced mask sampling active: original_train={len(train_dataset)}, "
                f"balanced_train={len(subset)}, buckets={len(groups)}, target_mode={target_mode}, "
                f"target_count={target_count}"
            )
            self.logger.info(f"Balanced mask bucket summary: {json.dumps(bucket_summary)}")
        return subset

    def _load_trials_status(self, ledger_path: Path, trial_ids: list[str], *, resume_trials: bool) -> dict[str, dict]:
        ledger: dict[str, dict] = {}
        if resume_trials and ledger_path.exists():
            try:
                payload = json.loads(ledger_path.read_text(encoding="utf-8"))
                if isinstance(payload, dict):
                    ledger = payload
            except Exception:
                ledger = {}

        for trial_id in trial_ids:
            entry = ledger.get(trial_id, {})
            if not isinstance(entry, dict):
                entry = {}
            entry.setdefault("status", "pending")
            entry.setdefault("started_at", None)
            entry.setdefault("ended_at", None)
            entry.setdefault("error", None)
            entry.setdefault("trial_row", None)
            ledger[trial_id] = entry
        return ledger

    def _save_trials_status(self, ledger_path: Path, ledger: dict[str, dict]):
        if not self.is_primary:
            return
        ledger_path.parent.mkdir(parents=True, exist_ok=True)
        ledger_path.write_text(json.dumps(ledger, indent=2), encoding="utf-8")

    def run(self):
        try:
            models_root = Path(self.paths.data.models)
            report_root = self.config_manager.get_stage_artifact_dir("S13")
            models_root.mkdir(parents=True, exist_ok=True)
            report_root.mkdir(parents=True, exist_ok=True)
            self._maybe_log_gpu_snapshot("Stage 12 start")

            resume_trials = bool(self._runtime_flag("resume_trials", True))
            run_final_pass_after_sweep = bool(self._runtime_flag("run_final_pass_after_sweep", True))

            if self.config.get("mock", False):
                dummy_best = models_root / "unet_best"
                dummy_final = models_root / "unet_final"
                dummy_best.mkdir(parents=True, exist_ok=True)
                dummy_final.mkdir(parents=True, exist_ok=True)
                (dummy_best / "model_index.json").write_text("{}", encoding="utf-8")
                (dummy_final / "model_index.json").write_text("{}", encoding="utf-8")
                pd.DataFrame({"epoch": [1, 2], "train_loss": [0.8, 0.5], "val_loss": [0.85, 0.55]}).to_csv(
                    report_root / "training_logs.csv", index=False
                )
                self.logger.info("=" * 20 + " STAGE 12 COMPLETED (MOCKED) " + "=" * 20 + "\\n")
                return

            dataset_root = Path(self.paths.data.inpainting)
            base_train_dataset = InpaintingDataset(dataset_root, self.dp_config.image_size, "train")
            val_dataset = InpaintingDataset(dataset_root, self.dp_config.image_size, "validation")
            train_dataset = base_train_dataset
            if bool(self._runtime_flag("balanced_mask_sampling", True)):
                train_dataset = self._build_balanced_train_subset(base_train_dataset, report_root)

            fallback_hparams = self._load_fallback_hyperparams()
            sweep_trials = self._load_sweep_trials()

            selected_hparams = fallback_hparams
            selected_trial_id = None

            if sweep_trials:
                trial_rows: list[dict] = []
                sweep_dir = report_root / "sweep"
                ledger_path = sweep_dir / "trials_status.json"
                trial_ids = [str(t.get("trial_id", "trial_unknown")) for t in sweep_trials]
                ledger = self._load_trials_status(ledger_path, trial_ids, resume_trials=resume_trials)
                self._save_trials_status(ledger_path, ledger)

                for trial in sweep_trials:
                    trial_id = str(trial.get("trial_id", "trial_unknown"))
                    trial_model_dir = models_root / "sweep" / trial_id
                    trial_report_dir = report_root / "sweep" / trial_id
                    entry = ledger.get(trial_id, {})
                    previous_status = entry.get("status")
                    previous_row = entry.get("trial_row")

                    if resume_trials and previous_status == "success" and isinstance(previous_row, dict):
                        self.logger.info(f"Stage S13 sweep: skipping successful trial '{trial_id}' (resume).")
                        trial_rows.append(previous_row)
                        continue

                    entry["status"] = "running"
                    entry["started_at"] = self._now_iso()
                    entry["ended_at"] = None
                    entry["error"] = None
                    self._save_trials_status(ledger_path, ledger)

                    try:
                        summary = self._run_training_job(
                            trial,
                            trial_model_dir,
                            trial_report_dir,
                            train_dataset,
                            val_dataset,
                            job_label=f"mini_sweep:{trial_id}",
                        )
                        row = {
                            "trial_id": trial_id,
                            "best_val_loss": summary.get("best_val_loss"),
                            "best_epoch": summary.get("best_epoch"),
                            "epochs_ran": summary.get("epochs_ran"),
                            "duration_seconds": summary.get("duration_seconds"),
                            "model_dir": str(trial_model_dir),
                            "report_dir": str(trial_report_dir),
                            **trial,
                        }
                        trial_rows.append(row)
                        entry["status"] = "success"
                        entry["trial_row"] = row
                        entry["ended_at"] = self._now_iso()
                    except Exception as exc:
                        entry["status"] = "failed"
                        entry["error"] = str(exc)
                        entry["ended_at"] = self._now_iso()
                        self._save_trials_status(ledger_path, ledger)
                        raise
                    self._save_trials_status(ledger_path, ledger)

                best = self._select_best_trial(trial_rows)
                if best is None:
                    raise RuntimeError("Mini-sweep ran but no trial summaries were produced.")

                selected_trial_id = best["trial_id"]
                selected_hparams = {
                    k: v
                    for k, v in best.items()
                    if k
                    in {
                        "learning_rate",
                        "train_batch_size",
                        "num_train_epochs",
                        "gradient_accumulation_steps",
                        "lr_scheduler",
                        "lr_warmup_steps",
                        "adam_weight_decay",
                        "max_grad_norm",
                        "early_stopping_patience",
                    }
                }

                stage11_best = self.config_manager.get_stage_artifact_path("S12", "best_hyperparameters.yaml")
                best_yaml = {
                    "selection_method": "mini_sweep_selected",
                    "selected_trial": selected_trial_id,
                    "selection_metric": "best_val_loss",
                    "best_val_loss": float(best.get("best_val_loss", 0.0)),
                    **selected_hparams,
                }
                if self.is_primary:
                    save_yaml(stage11_best, best_yaml)

                sweep_summary_csv = report_root / "sweep" / "trials_summary.csv"
                sweep_summary_json = report_root / "sweep" / "trials_summary.json"
                if self.is_primary:
                    sweep_summary_csv.parent.mkdir(parents=True, exist_ok=True)
                    pd.DataFrame(trial_rows).to_csv(sweep_summary_csv, index=False)
                    sweep_summary_json.write_text(json.dumps(trial_rows, indent=2), encoding="utf-8")
                    self._write_conditioning_interaction_report(report_root, trial_rows)

                if run_final_pass_after_sweep:
                    final_pass_summary_path = report_root / "final_pass_summary.json"
                    final_pass_done = (
                        resume_trials
                        and final_pass_summary_path.exists()
                        and (models_root / "unet_best").exists()
                        and (models_root / "unet_final").exists()
                    )
                    if final_pass_done:
                        self.logger.info("Stage S13 final-pass training already complete; skipping (resume).")
                    else:
                        self.logger.info(
                            f"Stage S13 final-pass: training selected trial '{selected_trial_id}' as deterministic output."
                        )
                        final_summary = self._run_training_job(
                            selected_hparams,
                            models_root,
                            report_root,
                            train_dataset,
                            val_dataset,
                            job_label=f"selected_full_run:{selected_trial_id}",
                        )
                        if self.is_primary:
                            final_pass_summary_path.write_text(
                                json.dumps(final_summary, indent=2),
                                encoding="utf-8",
                            )
                else:
                    best_model_dir = Path(best["model_dir"]) / "unet_best"
                    best_final_dir = Path(best["model_dir"]) / "unet_final"
                    if self.is_primary:
                        if best_model_dir.exists():
                            self._copy_model_dir(best_model_dir, models_root / "unet_best")
                        if best_final_dir.exists():
                            self._copy_model_dir(best_final_dir, models_root / "unet_final")

                    best_logs = Path(best["report_dir"]) / "training_logs.csv"
                    if self.is_primary and best_logs.exists():
                        shutil.copy2(best_logs, report_root / "training_logs.csv")

                self.logger.info(
                    f"Mini-sweep selected trial '{selected_trial_id}' "
                    f"(best_val_loss={best.get('best_val_loss')})."
                )
                self._generate_sweep_visuals(report_root, trial_rows)
            else:
                summary = self._run_training_job(
                    fallback_hparams,
                    models_root,
                    report_root,
                    train_dataset,
                    val_dataset,
                    job_label="single_run",
                )
                self.logger.info(
                    f"Single training run completed "
                    f"(best_val_loss={summary.get('best_val_loss')}, best_epoch={summary.get('best_epoch')})."
                )

            self._run_regime_comparison_experiment(
                models_root=models_root,
                report_root=report_root,
                hparams=selected_hparams,
                base_train_dataset=base_train_dataset,
                val_dataset=val_dataset,
            )

            # Final-model k-fold retraining with selected hyperparameters.
            kfold_cfg = self._to_plain_dict(self.config.get("kfold", {}))
            if bool(kfold_cfg.get("enabled", False)):
                n_splits = int(kfold_cfg.get("n_splits", 3))
                full_dataset = ConcatDataset([train_dataset, val_dataset])
                if len(full_dataset) >= n_splits:
                    fold_rows = []
                    kf = KFold(n_splits=n_splits, shuffle=True, random_state=self.global_seed)
                    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(range(len(full_dataset))), start=1):
                        fold_name = f"fold_{fold_idx:02d}"
                        fold_model_dir = models_root / "kfold" / fold_name
                        fold_report_dir = report_root / "kfold" / fold_name

                        train_subset = Subset(full_dataset, list(train_idx))
                        val_subset = Subset(full_dataset, list(val_idx))
                        summary = self._run_training_job(
                            selected_hparams,
                            fold_model_dir,
                            fold_report_dir,
                            train_subset,
                            val_subset,
                            job_label=f"kfold:{fold_name}",
                        )
                        fold_rows.append(
                            {
                                "fold": fold_name,
                                "train_size": len(train_subset),
                                "val_size": len(val_subset),
                                **summary,
                            }
                        )

                    summary_csv = report_root / "kfold_training_summary.csv"
                    summary_json = report_root / "kfold_training_summary.json"
                    if self.is_primary:
                        pd.DataFrame(fold_rows).to_csv(summary_csv, index=False)
                        summary_json.write_text(json.dumps(fold_rows, indent=2), encoding="utf-8")
                    self._generate_kfold_visuals(report_root, fold_rows)
                    self.logger.info(f"K-fold retraining completed ({n_splits} folds).")
                else:
                    self.logger.warning(
                        f"K-fold requested with n_splits={n_splits}, but dataset has only {len(full_dataset)} samples."
                    )

            self.logger.info("=" * 20 + " STAGE 12 COMPLETED " + "=" * 20 + "\\n")
        except Exception as exc:
            self.logger.exception(f"Error in Training: {exc}")
            raise


if __name__ == "__main__":
    cm = ConfigManager()
    stage = ModelTrainingStage(cm)
    stage.run()
