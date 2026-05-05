import logging
from pathlib import Path

import numpy as np
from PIL import Image

from thesis_pipeline.stages.stage_07_caption_generation import CaptionGenerationStage


def test_stage06b_grounding_reports_mask_metrics(tmp_path: Path):
    stage = CaptionGenerationStage.__new__(CaptionGenerationStage)
    stage.logger = logging.getLogger("test_stage06b_mask_metrics")
    stage.output_dir = tmp_path
    stage.filtered_dir = tmp_path
    stage.caption_cfg = {
        "grounding_validation": {
            "min_quadrant_macro_f1": 0.0,
            "min_border_touch_accuracy": 0.0,
            "min_area_correlation": -1.0,
            "strict_fail": False,
        }
    }
    stage.config = {"pipeline": {"strict_fail_policy": True}}

    mask_dir = tmp_path / "masks"
    mask_dir.mkdir(parents=True, exist_ok=True)

    m1 = np.zeros((32, 32), dtype=np.uint8)
    m1[0:8, 0:8] = 255
    Image.fromarray(m1).save(mask_dir / "wiki_001.png")

    m2 = np.zeros((32, 32), dtype=np.uint8)
    m2[20:30, 20:30] = 255
    Image.fromarray(m2).save(mask_dir / "eur_001.png")

    stage.filtered_dir = tmp_path
    (tmp_path / "masks").mkdir(exist_ok=True)

    report = stage._run_grounding_validation(
        raw_captions={"wiki_001.jpg": "raw", "eur_001.jpg": "raw"},
        enriched_captions={"wiki_001.jpg": "enriched", "eur_001.jpg": "enriched"},
        spatial_captions={
            "wiki_001.jpg": "damage on upper left rim edge",
            "eur_001.jpg": "damage on lower right body",
        },
    )

    assert report["sample_count"] == 2
    assert "quadrant_macro_f1" in report
    assert "border_touch_accuracy" in report
    assert "area_correlation" in report
    assert "mask_swap_delta" in report


