import pandas as pd

from thesis_pipeline.stages.stage_13_model_training import ModelTrainingStage


def test_pairwise_interactions_returns_effect_rows():
    df = pd.DataFrame(
        {
            "best_val_loss": [0.30, 0.25, 0.40, 0.35],
            "conditioning_finetune_mode": ["A", "A", "B", "B"],
            "mask_aware_loss": [True, False, True, False],
        }
    )

    rows = ModelTrainingStage._compute_pairwise_interactions(
        df,
        target_col="best_val_loss",
        factors=["conditioning_finetune_mode", "mask_aware_loss"],
    )

    assert rows
    assert all("interaction_effect" in r for r in rows)
    assert all("abs_interaction_effect" in r for r in rows)


def test_pairwise_interactions_empty_when_target_missing():
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    rows = ModelTrainingStage._compute_pairwise_interactions(
        df,
        target_col="best_val_loss",
        factors=["a", "b"],
    )
    assert rows == []


