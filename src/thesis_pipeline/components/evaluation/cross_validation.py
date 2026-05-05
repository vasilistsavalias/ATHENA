# src/thesis_pipeline/components/evaluation/cross_validation.py
"""3-Fold Cross-Validation for inpainting evaluation.

Provides confidence intervals on metric estimates by splitting test data
into k folds and evaluating on each fold independently.  The per-fold
results are then aggregated to produce mean ± std across folds.

Usage is integrated into Stage 13: after the primary evaluation on the
full test set, the cross-validator re-partitions the paired data and
reports fold-level statistics.
"""
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from sklearn.model_selection import KFold
from typing import List, Tuple, Dict

logger = logging.getLogger(__name__)


class CrossValidator:
    """K-fold cross-validation wrapper for inpainting evaluation.

    Parameters
    ----------
    k : int
        Number of folds (default 3).
    random_state : int
        Seed for reproducible fold assignment.
    """

    def __init__(self, k: int = 3, random_state: int = 42):
        self.k = k
        self.random_state = random_state

    def create_folds(
        self,
        paired_samples: List[Tuple[Path, Path]],
    ) -> List[List[Tuple[Path, Path]]]:
        """Split paired (image, mask) samples into k folds.

        Returns a list of k lists, each containing the samples for that fold.
        """
        kf = KFold(n_splits=self.k, shuffle=True, random_state=self.random_state)
        indices = np.arange(len(paired_samples))
        folds = []
        for _, test_idx in kf.split(indices):
            folds.append([paired_samples[i] for i in test_idx])
        return folds

    @staticmethod
    def aggregate_fold_results(
        fold_dataframes: List[pd.DataFrame],
    ) -> pd.DataFrame:
        """Aggregate per-fold metric DataFrames into mean ± std per model × metric.

        Parameters
        ----------
        fold_dataframes : list of DataFrames
            Each DF has columns: model, condition, sample_id, psnr, ssim, lpips, color, pattern, ...

        Returns
        -------
        pd.DataFrame
            Columns: model, metric, fold_1, fold_2, fold_3, mean, std, cv (coefficient of variation).
        """
        metrics = ['psnr', 'ssim', 'lpips', 'color', 'pattern']

        rows = []
        # Only consider Unconditional condition for CV analysis
        for metric in metrics:
            fold_means: Dict[str, List[float]] = {}
            for fold_idx, fold_df in enumerate(fold_dataframes):
                uncond = fold_df[fold_df['condition'] == 'Unconditional'] if 'condition' in fold_df.columns else fold_df
                per_model = uncond.groupby('model')[metric].mean()
                for model, val in per_model.items():
                    fold_means.setdefault(model, []).append(val)

            for model, vals in fold_means.items():
                row = {
                    'model': model,
                    'metric': metric,
                }
                for i, v in enumerate(vals):
                    row[f'fold_{i + 1}'] = round(v, 4)
                row['mean'] = round(np.mean(vals), 4)
                row['std'] = round(np.std(vals, ddof=1) if len(vals) > 1 else 0.0, 4)
                row['cv'] = round(row['std'] / row['mean'] * 100, 2) if row['mean'] != 0 else 0.0
                rows.append(row)

        return pd.DataFrame(rows)
