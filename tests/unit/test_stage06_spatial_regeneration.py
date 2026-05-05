import json
import logging
from pathlib import Path

import pytest

from thesis_pipeline.stages.stage_07_caption_generation import CaptionGenerationStage


def _build_stage(tmp_path: Path, *, strict_fail: bool) -> CaptionGenerationStage:
    stage = CaptionGenerationStage.__new__(CaptionGenerationStage)
    stage.logger = logging.getLogger("test_stage06_spatial_regeneration")
    stage.output_dir = tmp_path
    stage.caption_cfg = {
        "mock": True,
        "spatial_regeneration": {
            "enabled": True,
            "max_attempts": 2,
            "strict_fail": strict_fail,
            "fallback_prompt": "Describe only spatial damage.",
        },
    }
    stage.config = {"pipeline": {"strict_fail_policy": True}}
    return stage


def test_spatial_regeneration_writes_clean_file_and_report(tmp_path: Path):
    stage = _build_stage(tmp_path, strict_fail=False)
    image_path = tmp_path / "sample_a.jpg"
    image_path.write_bytes(b"dummy")

    cleaned = stage._run_spatial_regeneration_loop(
        [image_path],
        {"sample_a.jpg": "Beautiful museum masterpiece vase."},
    )

    assert cleaned == {}
    clean_file = tmp_path / "captions_spatial_clean.json"
    report_file = tmp_path / "caption_spatial_regeneration_report.json"
    assert clean_file.exists()
    assert report_file.exists()

    report = json.loads(report_file.read_text(encoding="utf-8"))
    assert report["unresolved_count"] == 1
    assert report["clean_count"] == 0
    assert "max_contamination_ratio" in report
    assert report["events"]
    first = report["events"][0]
    assert "contamination_ratio" in first
    assert "semantic_hit_count" in first
    assert "spatial_hit_count" in first


def test_spatial_regeneration_strict_fail_raises_on_unresolved(tmp_path: Path):
    stage = _build_stage(tmp_path, strict_fail=True)
    image_path = tmp_path / "sample_b.jpg"
    image_path.write_bytes(b"dummy")

    with pytest.raises(RuntimeError, match="Spatial regeneration failed"):
        stage._run_spatial_regeneration_loop(
            [image_path],
            {"sample_b.jpg": "Gorgeous vessel."},
        )


