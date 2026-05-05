import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from thesis_pipeline.stages.stage_13_model_training import ModelTrainingStage


def test_regime_comparison_writes_distinct_track_report(tmp_path: Path):
    stage = ModelTrainingStage.__new__(ModelTrainingStage)
    stage.is_primary = True
    stage.global_seed = 42
    stage.runtime_cfg = {
        "regime_comparison": {
            "enabled": True,
            "compare_metric": "best_val_loss",
            "strict_fail": False,
            "biased_regime_overrides": {"conditioning_finetune_mode": "cross_attention_only"},
            "balanced_regime_overrides": {"conditioning_finetune_mode": "cross_attention_plus_output"},
        }
    }
    stage.config_manager = SimpleNamespace(config={"pipeline": {"strict_fail_policy": True}})

    def fake_balanced_subset(dataset, _report_root):
        return ["balanced", *dataset]

    def fake_run_training_job(_hparams, model_dir, _report_dir, train_dataset, _val_dataset, *, job_label):
        (Path(model_dir) / "unet_best").mkdir(parents=True, exist_ok=True)
        score = 0.40 if "biased_regime" in job_label else 0.30
        return {
            "best_val_loss": score,
            "best_epoch": 1,
            "epochs_ran": 1,
            "duration_seconds": 0.01,
            "train_size_seen": len(train_dataset),
        }

    stage._build_balanced_train_subset = fake_balanced_subset
    stage._run_training_job = fake_run_training_job

    report = stage._run_regime_comparison_experiment(
        models_root=tmp_path / "models",
        report_root=tmp_path / "reports",
        hparams={"learning_rate": 1e-5},
        base_train_dataset=["a", "b"],
        val_dataset=["v"],
    )

    assert report is not None
    assert report["distinct_checkpoint_roots"] is True
    assert report["metric_delta_balanced_minus_biased"] == pytest.approx(-0.1)

    report_path = tmp_path / "reports" / "interaction_analysis" / "regime_comparison_report.json"
    assert report_path.exists()
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["biased_regime"]["model_dir"] != payload["balanced_regime"]["model_dir"]


