import logging
import os
from pathlib import Path

import pandas as pd
import pytest

if os.environ.get("TORCH_IMPORT_OK", "1") != "1":
    pytest.skip(
        "Skipping torch-dependent tests because torch import failed/crashed in a subprocess probe. "
        "Fix your torch install or set TORCH_IMPORT_OK=1 to force-run.",
        allow_module_level=True,
    )

pytest.importorskip("torch")

from thesis_pipeline.stages.stage_15_model_evaluation import ModelEvaluationStage


def _build_stage() -> ModelEvaluationStage:
    stage = ModelEvaluationStage.__new__(ModelEvaluationStage)
    stage.logger = logging.getLogger("test_stage13_all_vs_all")
    return stage


def _sample_df() -> pd.DataFrame:
    rows = []
    for sample_id in [f"s{i}" for i in range(6)]:
        rows.extend(
            [
                {
                    "model": "Telea",
                    "condition": "Unconditional",
                    "sample_id": sample_id,
                    "mask_type": "rect",
                    "mask_coverage": 0.12,
                    "psnr": 15.0,
                    "ssim": 0.60,
                    "lpips": 0.40,
                    "color": 0.16,
                    "pattern": 0.80,
                },
                {
                    "model": "FT-SD",
                    "condition": "Unconditional",
                    "sample_id": sample_id,
                    "mask_type": "rect",
                    "mask_coverage": 0.12,
                    "psnr": 18.0,
                    "ssim": 0.67,
                    "lpips": 0.31,
                    "color": 0.20,
                    "pattern": 0.89,
                },
                {
                    "model": "FT-SD",
                    "condition": "Enriched Text",
                    "sample_id": sample_id,
                    "mask_type": "rect",
                    "mask_coverage": 0.12,
                    "psnr": 18.4,
                    "ssim": 0.69,
                    "lpips": 0.29,
                    "color": 0.21,
                    "pattern": 0.90,
                },
            ]
        )
    return pd.DataFrame(rows)


def test_run_stats_all_vs_all_exports_ci_columns(tmp_path: Path):
    stage = _build_stage()
    df = _sample_df()
    out_dir = tmp_path / "stats"
    out_dir.mkdir(parents=True, exist_ok=True)

    stats_df = stage._run_stats(df, out_dir)

    assert not stats_df.empty
    assert (out_dir / "paired_t_tests.csv").exists()
    assert {"comparison_type", "model_a", "model_b", "ci_95_lower", "ci_95_upper"}.issubset(stats_df.columns)
    assert set(stats_df["comparison_type"].unique()) == {"all_vs_all"}
    assert stats_df["comparison"].nunique() == 3


def test_error_analysis_exports_mask_and_severity_tables(tmp_path: Path):
    stage = _build_stage()
    df = _sample_df()
    df.loc[df["sample_id"].isin(["s4", "s5"]), "mask_type"] = "irregular"
    df.loc[df["sample_id"].isin(["s4", "s5"]), "mask_coverage"] = 0.52
    out_dir = tmp_path / "stats"
    out_dir.mkdir(parents=True, exist_ok=True)

    stage._save_error_analysis(df, out_dir)

    assert (out_dir / "error_by_mask_type.csv").exists()
    assert (out_dir / "error_by_severity.csv").exists()
    assert (out_dir / "error_by_mask_type_and_severity.csv").exists()

    severity_df = pd.read_csv(out_dir / "error_by_severity.csv")
    assert "severity_bin" in severity_df.columns
    assert len(severity_df) >= 2


def test_caption_coverage_report_counts_only_usable_prompt_text():
    stage = _build_stage()
    report = stage._build_caption_coverage_report(
        {"sample_a.jpg", "sample_b.jpg", "sample_c.jpg"},
        raw_caps={
            "sample_a.jpg": "metadata text",
            "sample_b.jpg": "   ",
        },
        enriched_caps={
            "sample_a.jpg": "dense caption",
            "sample_b.jpg": "...",
            "sample_c.jpg": "enriched caption",
        },
        refined_caps={
            "sample_a.jpg": "refined caption",
            "sample_b.jpg": "refined caption",
            "sample_c.jpg": "",
        },
        threshold=0.5,
    )

    assert report["families"]["raw"]["usable_count"] == 1
    assert report["families"]["raw"]["empty_count"] == 1
    assert report["families"]["raw"]["missing_count"] == 1
    assert report["families"]["enriched"]["usable_count"] == 2
    assert report["families"]["refined"]["usable_count"] == 2
    assert report["prompt_ready_sample_keys"] == ["sample_a.jpg"]


