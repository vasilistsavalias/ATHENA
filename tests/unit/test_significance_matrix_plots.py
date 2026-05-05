from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from thesis_pipeline.visualization.plots import ThesisPlotter


def _build_stats_for_single_pair() -> pd.DataFrame:
    rows = []
    metrics = ["psnr", "ssim", "lpips", "color", "pattern"]
    for metric in metrics:
        rows.append(
            {
                "comparison": "LaMa [Unconditional] vs Telea [Unconditional]",
                "comparison_type": "all_vs_all",
                "model_a": "LaMa",
                "condition_a": "Unconditional",
                "model_b": "Telea",
                "condition_b": "Unconditional",
                "metric": metric,
                "cohens_d": 0.42 if metric != "lpips" else -0.30,
                "significant_bonferroni": True,
            }
        )
        rows.append(
            {
                "comparison": "LaMa [Raw Text] vs Telea [Unconditional]",
                "comparison_type": "all_vs_all",
                "model_a": "LaMa",
                "condition_a": "Raw Text",
                "model_b": "Telea",
                "condition_b": "Unconditional",
                "metric": metric,
                "cohens_d": 0.20 if metric != "lpips" else -0.10,
                "significant_bonferroni": False,
            }
        )
    return pd.DataFrame(rows)


def test_significance_pair_builder_counts():
    pairs = ThesisPlotter.build_significance_matrix_pairs()
    assert "big_vs_base" in pairs
    assert "big_four_internal" in pairs
    assert len(pairs["big_vs_base"]) == 15
    assert len(pairs["big_four_internal"]) == 6


def test_significance_scope_filter_and_manifest(tmp_path: Path):
    plotter = ThesisPlotter(tmp_path / "charts")
    stats_df = _build_stats_for_single_pair()

    effect_uncond, state_uncond = plotter._extract_pair_scope_tables(
        stats_df,
        model_anchor="LaMa",
        model_other="Telea",
        scope="unconditional",
    )
    assert len(effect_uncond.index) == 1
    assert "Unconditional vs Unconditional" in effect_uncond.index
    assert len(state_uncond.index) == 1

    effect_expanded, state_expanded = plotter._extract_pair_scope_tables(
        stats_df,
        model_anchor="LaMa",
        model_other="Telea",
        scope="condition_expanded",
    )
    assert len(effect_expanded.index) == 2
    assert "Raw Text vs Unconditional" in effect_expanded.index
    assert len(state_expanded.index) == 2

    manifest_df = plotter.plot_significance_matrix_suite(
        stats_df=stats_df,
        config={
            "enabled": True,
            "scopes": ["unconditional"],
            "families": ["big_vs_base"],
            "output_subdir": "significance_matrix",
        },
    )

    # 15 pairs * 1 scope * 2 chart types
    assert len(manifest_df) == 30
    generated = manifest_df[manifest_df["status"] == "generated"]
    skipped = manifest_df[manifest_df["status"] == "skipped"]
    assert len(generated) == 2
    assert len(skipped) == 28

    for path_str in generated["filepath"].tolist():
        assert Path(path_str).exists()

    assert (tmp_path / "charts" / "significance_matrix" / "matrix_manifest.csv").exists()
    assert (tmp_path / "charts" / "significance_matrix" / "matrix_manifest.json").exists()


def test_main_config_defaults_to_full_test_and_has_significance_matrix():
    cfg = yaml.safe_load(Path("config/pipeline/main_config.yaml").read_text(encoding="utf-8"))
    model_eval = cfg["model_evaluation"]
    assert model_eval["phase"] == "full_test"
    matrix_cfg = model_eval["significance_matrix"]
    assert matrix_cfg["enabled"] is True
    assert matrix_cfg["scopes"] == ["unconditional", "condition_expanded"]
    assert matrix_cfg["families"] == ["big_vs_base", "big_four_internal"]
