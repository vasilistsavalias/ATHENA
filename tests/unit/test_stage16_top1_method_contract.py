import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from thesis_pipeline.stages.stage_18_expert_validation import ExpertValidationStage


class DummyConfig(dict):
    def __init__(self):
        super().__init__({"expert_validation": {}})
        self.global_params = {"random_state": 42}


class DummyConfigManager:
    def __init__(self, stage_13_dir: Path, inpainting_dir: Path):
        self.config = DummyConfig()
        self._stage_15 = stage_13_dir
        self._stage_18 = stage_13_dir / "stage18"
        self._stage_02 = stage_13_dir.parent / "stage02"
        self._paths = SimpleNamespace(
            artifacts=SimpleNamespace(
                stage_13=str(stage_13_dir),
                stage_16=str(stage_13_dir / "stage16"),
            ),
            data=SimpleNamespace(inpainting=str(inpainting_dir)),
        )

    def get_paths(self):
        return self._paths

    def get_stage_artifact_dir(self, stage_id: str) -> Path:
        if stage_id == "S02":
            return self._stage_02
        if stage_id == "S15":
            return self._stage_15
        if stage_id == "S18":
            return self._stage_18
        raise KeyError(stage_id)

    def get_stage_artifact_path(self, stage_id: str, *parts: str) -> Path:
        return self.get_stage_artifact_dir(stage_id).joinpath(*parts)


def build_stage(tmp_path: Path) -> ExpertValidationStage:
    stage_13_dir = tmp_path / "stage13"
    inpainting_dir = tmp_path / "inpainting"
    (stage_13_dir / "benchmarking_matrix").mkdir(parents=True)
    manager = DummyConfigManager(stage_13_dir, inpainting_dir)
    return ExpertValidationStage(manager)


def write_top1_payload(stage: ExpertValidationStage, payload: dict) -> None:
    top1_json = stage.eval_root / "benchmarking_matrix" / "top1_method.json"
    top1_json.write_text(json.dumps(payload), encoding="utf-8")


def test_load_top1_method_accepts_historical_model_key(tmp_path: Path):
    stage = build_stage(tmp_path)
    write_top1_payload(
        stage,
        {
            "model": "FT-SD+TTA",
            "condition": "Unconditional",
            "composite_rank": 1.33,
            "composite_score": 11.67,
        },
    )

    assert stage._load_top1_method() == "FT-SD+TTA"


def test_load_top1_method_rejects_payload_without_method_key(tmp_path: Path):
    stage = build_stage(tmp_path)
    write_top1_payload(stage, {"condition": "Unconditional"})

    with pytest.raises(ValueError, match="missing a method key"):
        stage._load_top1_method()


def test_find_ground_truth_uses_stage13_original_png(tmp_path: Path):
    stage = build_stage(tmp_path)
    sample_dir = stage.samples_dir / "sample_a"
    sample_dir.mkdir(parents=True)
    original_png = sample_dir / "original.png"
    original_png.write_bytes(b"png")

    assert stage._find_ground_truth("sample_a.png") == original_png


def test_find_ground_truth_uses_test_split_ground_truth_dir(tmp_path: Path):
    stage = build_stage(tmp_path)
    stage.test_gt_dir.mkdir(parents=True)
    gt_png = stage.test_gt_dir / "sample_b.png"
    gt_png.write_bytes(b"png")

    assert stage._find_ground_truth("sample_b.png") == gt_png


