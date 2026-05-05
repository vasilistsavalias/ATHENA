"""S14 - Deep baseline fine-tuning stage."""

import json
import logging
from pathlib import Path
from typing import Any

from thesis_pipeline.config_manager import ConfigManager
from thesis_pipeline.components.baseline_finetuning.adapter import (
    AdapterRegistry,
    FTResult,
)


class BaselineFineTuningStage:
    """Pipeline stage S14: baseline model domain fine-tuning."""

    def __init__(self, config_manager: ConfigManager):
        self.config_manager = config_manager
        self.paths = config_manager.get_paths()
        self.logger = logging.getLogger(__name__)

        self.ft_cfg: dict[str, Any] = dict(config_manager.config.get("baseline_finetuning", {}))
        self.enabled = bool(self.ft_cfg.get("enabled", True))
        self.models: list[str] = list(self.ft_cfg.get("models", ["LaMa", "MAT", "CoModGAN"]))
        self.epochs = int(self.ft_cfg.get("epochs", 50))
        self.batch_size = int(self.ft_cfg.get("batch_size", 8))
        self.learning_rate = float(self.ft_cfg.get("learning_rate", 1e-4))
        self.early_stopping_patience = int(self.ft_cfg.get("early_stopping_patience", 10))
        self.device = str(self.ft_cfg.get("device", "cuda"))
        self.checkpoint_dir = Path(
            self.ft_cfg.get("checkpoint_dir",
                            str(Path(self.paths.data.models) / "baselines"))
        )
        self.data_root = Path(self.paths.data.inpainting)

        # Output directory for reports
        self.output_dir = self.config_manager.get_stage_artifact_dir("S14")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(self) -> None:
        self.logger.info("=" * 20 + " STAGE S14: Deep Baseline Fine-Tuning " + "=" * 20)

        if not self.enabled:
            self.logger.info("S14 disabled via config. Skipping.")
            return

        if not self.data_root.exists():
            raise RuntimeError(f"S14: inpainting data root does not exist: {self.data_root}")

        results: list[dict] = []

        for model_name in self.models:
            self.logger.info(f"--- Fine-tuning {model_name} ---")
            try:
                AdapterClass = AdapterRegistry.get(model_name)
            except ValueError:
                self.logger.error(f"S14: unknown baseline model: {model_name}")
                continue

            model_cfg = dict(self.ft_cfg.get(model_name.lower(), {}))
            adapter = AdapterClass(
                cfg=model_cfg,
                data_root=self.data_root,
                checkpoint_dir=self.checkpoint_dir,
                device=self.device,
            )

            try:
                adapter.setup()
                result: FTResult = adapter.train(
                    epochs=self.epochs,
                    batch_size=self.batch_size,
                    learning_rate=self.learning_rate,
                    early_stopping_patience=self.early_stopping_patience,
                )
                ckpt = adapter.export()
                self.logger.info(
                    f"  {model_name}: best_epoch={result.best_epoch}, "
                    f"val_loss={result.best_val_loss:.5f}, ckpt={ckpt}"
                )
                results.append({
                    "model": model_name,
                    "status": "success",
                    "best_epoch": result.best_epoch,
                    "best_val_loss": result.best_val_loss,
                    "total_epochs": result.total_epochs,
                    "checkpoint": str(ckpt),
                    "metrics": result.metrics,
                })
            except Exception as e:
                self.logger.error(f"  {model_name}: fine-tuning failed: {e}", exc_info=True)
                results.append({
                    "model": model_name,
                    "status": "failed",
                    "error": str(e),
                })

        # Write summary
        summary_path = self.output_dir / "baseline_finetuning_summary.json"
        summary_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        self.logger.info(f"S14 summary written to {summary_path}")

        # Check if any required model failed
        failed = [r for r in results if r["status"] == "failed"]
        if failed:
            names = [r["model"] for r in failed]
            self.logger.warning(f"S14: {len(failed)} model(s) failed fine-tuning: {names}")
