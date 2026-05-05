from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from box import ConfigBox

if os.environ.get("TORCH_IMPORT_OK", "1") != "1":
    pytest.skip(
        "Skipping torch-dependent tests because torch import failed/crashed in a subprocess probe. "
        "Fix your torch install or set TORCH_IMPORT_OK=1 to force-run.",
        allow_module_level=True,
    )

pytest.importorskip("torch")

from thesis_pipeline.stages.stage_13_model_training import ModelTrainingStage


class _DummyConfigManager:
    def __init__(self, root: Path):
        self.root = root
        self._training = ConfigBox({"mock": False})
        self._stage_12 = root / "stage11"
        self._stage_13 = root / "stage12"
        self._paths = SimpleNamespace(
            data=SimpleNamespace(inpainting=str(root / "inpainting"), models=str(root / "models")),
            artifacts=SimpleNamespace(stage_11=str(root / "stage11"), stage_12=str(root / "stage12"), logs=str(root / "logs")),
        )
        self._dp = ConfigBox({"image_size": [512, 512]})
        self._global = ConfigBox({"random_state": 42})

    def get_training_config(self):
        return self._training

    def get_paths(self):
        return self._paths

    def get_data_processing_config(self):
        return self._dp

    def get_global_params(self):
        return self._global

    def get_stage_artifact_dir(self, stage_id: str) -> Path:
        if stage_id == "S12":
            return self._stage_12
        if stage_id == "S13":
            return self._stage_13
        raise KeyError(stage_id)

    def get_stage_artifact_path(self, stage_id: str, *parts: str) -> Path:
        return self.get_stage_artifact_dir(stage_id).joinpath(*parts)


def test_select_best_trial_uses_lowest_finite_loss(tmp_path):
    stage = ModelTrainingStage(_DummyConfigManager(tmp_path))
    best = stage._select_best_trial(
        [
            {"trial_id": "trial_01", "best_val_loss": 0.42},
            {"trial_id": "trial_02", "best_val_loss": float("nan")},
            {"trial_id": "trial_03", "best_val_loss": 0.31},
        ]
    )
    assert best is not None
    assert best["trial_id"] == "trial_03"


def test_load_sweep_trials_normalizes_trial_ids(tmp_path):
    cm = _DummyConfigManager(tmp_path)
    stage11 = Path(cm.get_paths().artifacts.stage_11)
    stage11.mkdir(parents=True, exist_ok=True)

    payload = {
        "selection_method": "mini_sweep",
        "trials": [
            {"learning_rate": 1e-5, "adam_weight_decay": 1e-2},
            {"trial_id": "custom", "learning_rate": 7e-6, "adam_weight_decay": 5e-3},
        ],
    }
    (stage11 / "sweep_plan.yaml").write_text(yaml.safe_dump(payload), encoding="utf-8")

    stage = ModelTrainingStage(cm)
    trials = stage._load_sweep_trials()

    assert len(trials) == 2
    assert trials[0]["trial_id"] == "trial_01"
    assert trials[1]["trial_id"] == "custom"


