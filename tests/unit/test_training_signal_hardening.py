from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
from box import ConfigBox
from PIL import Image

if os.environ.get("TORCH_IMPORT_OK", "1") != "1":
    pytest.skip(
        "Skipping torch-dependent tests because torch import failed/crashed in a subprocess probe. "
        "Fix your torch install or set TORCH_IMPORT_OK=1 to force-run.",
        allow_module_level=True,
    )

pytest.importorskip("torch")
from torch.utils.data import Subset

from thesis_pipeline.stages.stage_12_hyperparameter_tuning import HyperparameterTuningStage
from thesis_pipeline.stages.stage_13_model_training import ModelTrainingStage


class _DummyConfigManager:
    def __init__(self, root: Path):
        self.root = root
        self._training = ConfigBox({"mock": False})
        self._hyper = ConfigBox(
            {
                "mode": "mini_sweep",
                "learning_rate_range": [1e-6, 5e-5],
                "max_trials": 8,
                "mini_sweep": {
                    "lr_values": [5e-6, 7e-6, 1e-5, 1.2e-5],
                    "adam_weight_decay_values": [5e-3, 1e-2],
                    "lr_warmup_steps_values": [100, 300],
                    "max_grad_norm_values": [0.8, 1.0],
                    "early_stopping_patience": 3,
                },
            }
        )
        self._paths = SimpleNamespace(
            data=SimpleNamespace(inpainting=str(root / "inpainting"), models=str(root / "models")),
            artifacts=SimpleNamespace(
                stage_11=str(root / "stage11"),
                stage_12=str(root / "stage12"),
                logs=str(root / "logs"),
            ),
        )
        self._dp = ConfigBox({"image_size": [16, 16]})
        self._global = ConfigBox({"random_state": 42})
        self.config = ConfigBox(
            {
                "model_training": {
                    "balanced_mask_sampling": True,
                    "balanced_sampling_target": "median",
                    "balanced_sampling_area_thresholds": [0.12, 0.30],
                    "balanced_sampling_seed": 42,
                }
            }
        )

    def get_training_config(self):
        return self._training

    def get_hyperparameter_tuning_config(self):
        return self._hyper

    def get_paths(self):
        return self._paths

    def get_data_processing_config(self):
        return self._dp

    def get_global_params(self):
        return self._global


class _FakeDataset:
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)


def _write_mask(path: Path, ratio: float):
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("L", (10, 10), 0)
    white_pixels = int(round(100 * ratio))
    if white_pixels > 0:
        coords = [(idx % 10, idx // 10) for idx in range(min(white_pixels, 100))]
        for x, y in coords:
            image.putpixel((x, y), 255)
    image.save(path)


def test_stage11_mini_sweep_includes_warmup_and_gradnorm(tmp_path):
    stage = HyperparameterTuningStage(_DummyConfigManager(tmp_path))
    trials = stage._build_mini_sweep_trials(1e-6, 5e-5)

    assert len(trials) == 8
    assert all("lr_warmup_steps" in row for row in trials)
    assert all("max_grad_norm" in row for row in trials)
    assert len({float(row["learning_rate"]) for row in trials}) >= 2


def test_stage12_balanced_sampling_returns_subset(tmp_path):
    cm = _DummyConfigManager(tmp_path)
    stage = ModelTrainingStage(cm)
    report_root = Path(cm.get_paths().artifacts.stage_12)
    report_root.mkdir(parents=True, exist_ok=True)

    data_root = tmp_path / "masks"
    samples = []
    stems = [
        ("sample_irregular_a", 0.08),
        ("sample_rect_a", 0.20),
        ("sample_rect_b", 0.22),
        ("sample_edge_a", 0.45),
        ("sample_edge_b", 0.50),
        ("sample_edge_c", 0.55),
    ]
    for stem, ratio in stems:
        image_path = data_root / f"{stem}.png"
        mask_path = data_root / f"{stem}_mask.png"
        caption_path = data_root / f"{stem}.txt"
        _write_mask(mask_path, ratio)
        image_path.touch()
        caption_path.write_text("test", encoding="utf-8")
        samples.append((image_path, mask_path, caption_path))

    dataset = _FakeDataset(samples)
    subset = stage._build_balanced_train_subset(dataset, report_root)

    assert isinstance(subset, Subset)
    assert len(subset) == 6
    cache = json.loads((report_root / "mask_sampling_cache_train.json").read_text(encoding="utf-8"))
    assert "sample_irregular_a" in cache


