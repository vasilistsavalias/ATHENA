from __future__ import annotations

from pathlib import Path

import pandas as pd

from thesis_pipeline.visualization.plots import ThesisPlotter


def test_plot_spatial_contamination_distribution(tmp_path: Path):
    plotter = ThesisPlotter(tmp_path)
    report = {
        "max_contamination_ratio": 0.25,
        "events": [
            {"sample_id": "a", "contamination_ratio": 0.10},
            {"sample_id": "b", "contamination_ratio": 0.20},
            {"sample_id": "c", "contamination_ratio": 0.35},
            {"sample_id": "d", "contamination_ratio": 0.50},
        ],
    }

    plotter.plot_spatial_contamination_distribution(report)

    assert (tmp_path / "spatial_contamination_distribution.png").exists()


def test_plot_grounding_validation_panel(tmp_path: Path):
    plotter = ThesisPlotter(tmp_path)
    report = {
        "sample_count": 6,
        "pass": True,
        "quadrant_macro_f1": 0.82,
        "mask_swap_quadrant_macro_f1": 0.41,
        "min_quadrant_macro_f1": 0.60,
        "border_touch_accuracy": 0.86,
        "mask_swap_border_touch_accuracy": 0.48,
        "min_border_touch_accuracy": 0.75,
        "area_correlation": 0.65,
        "mask_swap_area_correlation": 0.11,
        "min_area_correlation": 0.40,
        "mask_swap_delta": {
            "quadrant_macro_f1": 0.41,
            "border_touch_accuracy": 0.38,
            "area_correlation": 0.54,
        },
    }

    plotter.plot_grounding_validation_panel(report)

    assert (tmp_path / "grounding_validation_panel.png").exists()


def test_plot_regime_source_interaction(tmp_path: Path):
    plotter = ThesisPlotter(tmp_path)
    interaction_df = pd.DataFrame(
        [
            {"regime": "biased", "source_split": "wikimedia", "delta_psnr": -0.21, "ci_low": -0.30, "ci_high": -0.12},
            {"regime": "biased", "source_split": "europeana", "delta_psnr": -0.14, "ci_low": -0.25, "ci_high": -0.04},
            {"regime": "biased", "source_split": "combined", "delta_psnr": -0.18, "ci_low": -0.28, "ci_high": -0.08},
            {"regime": "balanced", "source_split": "wikimedia", "delta_psnr": -0.06, "ci_low": -0.13, "ci_high": 0.01},
            {"regime": "balanced", "source_split": "europeana", "delta_psnr": -0.03, "ci_low": -0.11, "ci_high": 0.05},
            {"regime": "balanced", "source_split": "combined", "delta_psnr": -0.04, "ci_low": -0.10, "ci_high": 0.02},
        ]
    )

    plotter.plot_regime_source_interaction(interaction_df)

    assert (tmp_path / "regime_source_interaction.png").exists()


def test_plot_frozen_control_integrity_table(tmp_path: Path):
    plotter = ThesisPlotter(tmp_path)
    integrity_df = pd.DataFrame(
        [
            {
                "model": "Telea",
                "samples_hashed": 12,
                "unique_hashes": 12,
                "requires_grad": False,
                "trainable_params": 0,
                "status": "ok",
            },
            {
                "model": "Vanilla SD",
                "samples_hashed": 12,
                "unique_hashes": 12,
                "requires_grad": False,
                "trainable_params": 0,
                "status": "ok",
            },
        ]
    )

    plotter.plot_frozen_control_integrity_table(integrity_df)

    assert (tmp_path / "frozen_control_integrity_table.png").exists()


def test_plot_expert_reliability_heatmap(tmp_path: Path):
    plotter = ThesisPlotter(tmp_path)
    item_level_df = pd.DataFrame(
        [
            {
                "participant_id": "R0001",
                "item_id": 101,
                "block": "A",
                "is_attention_check": True,
                "authenticity_likelihood": 4,
                "archaeological_plausibility": 5,
            },
            {
                "participant_id": "R0001",
                "item_id": 102,
                "block": "A",
                "is_attention_check": True,
                "authenticity_likelihood": 3,
                "archaeological_plausibility": 4,
            },
            {
                "participant_id": "R0002",
                "item_id": 101,
                "block": "A",
                "is_attention_check": True,
                "authenticity_likelihood": 5,
                "archaeological_plausibility": 4,
            },
            {
                "participant_id": "R0002",
                "item_id": 102,
                "block": "A",
                "is_attention_check": True,
                "authenticity_likelihood": 2,
                "archaeological_plausibility": 3,
            },
        ]
    )
    reliability_payload = {
        "block_a": {
            "pairwise_within_onepoint_authenticity": 0.72,
            "pairwise_within_onepoint_plausibility": 0.68,
        }
    }

    plotter.plot_expert_reliability_heatmap(item_level_df, reliability_payload)

    assert (tmp_path / "expert_reliability_heatmap.png").exists()
