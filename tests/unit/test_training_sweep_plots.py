from __future__ import annotations

from pathlib import Path

import pandas as pd

from thesis_pipeline.visualization.plots import ThesisPlotter


def test_training_sweep_plot_generation(tmp_path):
    plotter = ThesisPlotter(tmp_path)

    sweep_df = pd.DataFrame(
        [
            {"trial_id": "trial_01", "best_val_loss": 0.32, "duration_seconds": 1200, "learning_rate": 7e-6, "adam_weight_decay": 5e-3},
            {"trial_id": "trial_02", "best_val_loss": 0.30, "duration_seconds": 1400, "learning_rate": 7e-6, "adam_weight_decay": 1e-2},
            {"trial_id": "trial_03", "best_val_loss": 0.29, "duration_seconds": 1500, "learning_rate": 1.2e-5, "adam_weight_decay": 1e-2},
        ]
    )

    plotter.plot_sweep_pareto(sweep_df)
    plotter.plot_lr_wd_response_heatmap(sweep_df)

    sweep_root = tmp_path / "sweep"
    for i in range(1, 4):
        tdir = sweep_root / f"trial_{i:02d}"
        tdir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            {
                "epoch": [1, 2, 3],
                "train_loss": [0.8 - i * 0.05, 0.6 - i * 0.05, 0.5 - i * 0.04],
                "val_loss": [0.85 - i * 0.05, 0.65 - i * 0.05, 0.55 - i * 0.04],
            }
        ).to_csv(tdir / "training_logs.csv", index=False)

    plotter.plot_trial_learning_curves_panel(sweep_root)

    kfold_df = pd.DataFrame(
        [
            {"fold": "fold_01", "best_val_loss": 0.30},
            {"fold": "fold_02", "best_val_loss": 0.32},
            {"fold": "fold_03", "best_val_loss": 0.31},
        ]
    )
    plotter.plot_kfold_training_stability(kfold_df)

    assert (tmp_path / "sweep_pareto.png").exists()
    assert (tmp_path / "lr_wd_response_heatmap.png").exists()
    assert (tmp_path / "trial_learning_curves_panel.png").exists()
    assert (tmp_path / "kfold_training_stability.png").exists()
