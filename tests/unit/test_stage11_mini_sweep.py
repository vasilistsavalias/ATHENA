from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import yaml
from box import ConfigBox

from thesis_pipeline.stages.stage_12_hyperparameter_tuning import HyperparameterTuningStage


class _DummyPlotter:
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)

    def plot_table(self, *args, **kwargs):
        return None

    def plot_hyperparameter_summary(self, *args, **kwargs):
        return None

    def plot_lr_schedule(self, *args, **kwargs):
        return None


class _DummyConfigManager:
    def __init__(self, stage11_dir: Path, hp_cfg: dict):
        self._hp_cfg = ConfigBox(hp_cfg)
        self._paths = SimpleNamespace(artifacts=SimpleNamespace(stage_11=str(stage11_dir)))

    def get_hyperparameter_tuning_config(self):
        return self._hp_cfg

    def get_paths(self):
        return self._paths


def test_stage11_writes_mini_sweep_plan(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "thesis_pipeline.stages.stage_12_hyperparameter_tuning.ThesisPlotter",
        _DummyPlotter,
    )

    cfg = {
        "mode": "mini_sweep",
        "learning_rate_range": [1e-6, 5e-5],
        "max_trials": 6,
        "mini_sweep": {
            "lr_values": [7e-6, 1.2e-5],
            "adam_weight_decay_values": [5e-3, 1e-2, 2e-2],
            "selection_metric": "best_val_loss",
            "early_stopping_patience": 3,
        },
    }
    stage = HyperparameterTuningStage(_DummyConfigManager(tmp_path, cfg))
    stage.run()

    sweep_yaml = tmp_path / "sweep_plan.yaml"
    sweep_csv = tmp_path / "sweep_plan.csv"
    best_yaml = tmp_path / "best_hyperparameters.yaml"

    assert sweep_yaml.exists()
    assert sweep_csv.exists()
    assert best_yaml.exists()

    payload = yaml.safe_load(sweep_yaml.read_text(encoding="utf-8"))
    assert payload["selection_method"] == "mini_sweep"
    assert len(payload["trials"]) == 6

    df = pd.read_csv(sweep_csv)
    assert len(df) == 6
    assert df["trial_id"].is_unique

    best = yaml.safe_load(best_yaml.read_text(encoding="utf-8"))
    assert best["selection_method"] == "mini_sweep_pending_stage12"
    assert best["selected_trial"].startswith("trial_")


def test_stage11_caps_trials_by_max_trials(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "thesis_pipeline.stages.stage_12_hyperparameter_tuning.ThesisPlotter",
        _DummyPlotter,
    )

    cfg = {
        "mode": "mini_sweep",
        "learning_rate_range": [1e-6, 5e-5],
        "max_trials": 4,
        "mini_sweep": {
            "lr_values": [1e-6, 7e-6, 1.2e-5],
            "adam_weight_decay_values": [5e-3, 1e-2],
            "selection_metric": "best_val_loss",
            "early_stopping_patience": 3,
        },
    }
    stage = HyperparameterTuningStage(_DummyConfigManager(tmp_path, cfg))
    stage.run()

    df = pd.read_csv(tmp_path / "sweep_plan.csv")
    assert len(df) == 4


