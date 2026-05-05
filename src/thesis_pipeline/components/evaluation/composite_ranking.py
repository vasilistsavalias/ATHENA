"""
Composite ranking for the unified evaluation matrix.

Reads the benchmarking matrix CSV produced by S08 and computes a single
composite score per (model, condition) combination using rank-averaging
across all metrics.  The top-1 method is exported for downstream use
by S10 (expert validation in Top-1 vs Real mode).

Metric directions:
    PSNR ↑  SSIM ↑  LPIPS ↓  color_fidelity ↑  pattern_preservation ↑
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Metrics and direction (True = higher is better)
METRIC_DIRECTIONS: dict[str, bool] = {
    "psnr": True,
    "ssim": True,
    "lpips": False,
    "color_fidelity": True,
    "pattern_preservation": True,
}

# Default weights (equal)
DEFAULT_WEIGHTS: dict[str, float] = {k: 1.0 for k in METRIC_DIRECTIONS}


def compute_composite_ranking(
    df: pd.DataFrame,
    *,
    metrics: dict[str, bool] | None = None,
    weights: dict[str, float] | None = None,
    group_cols: tuple[str, ...] = ("model", "condition"),
) -> pd.DataFrame:
    """Compute composite ranking from the benchmarking matrix.

    Parameters
    ----------
    df : DataFrame
        The raw per-sample evaluation matrix.  Must contain columns in
        *group_cols* plus at least one of the metric columns.
    metrics : dict[str, bool] | None
        Metric name → higher_is_better.  Defaults to ``METRIC_DIRECTIONS``.
    weights : dict[str, float] | None
        Per-metric weight.  Defaults to equal weights.
    group_cols : tuple[str, ...]
        Columns that define a method variant.

    Returns
    -------
    DataFrame
        One row per (model, condition) with columns:
        ``model, condition, <metric>_mean, <metric>_rank, composite_rank,
        composite_score``.
    """
    if metrics is None:
        metrics = METRIC_DIRECTIONS
    if weights is None:
        weights = DEFAULT_WEIGHTS

    # Filter to available metrics in the dataframe
    available = [m for m in metrics if m in df.columns]
    if not available:
        raise ValueError(f"None of {list(metrics)} found in dataframe columns: {list(df.columns)}")

    # Compute group means
    agg = df.groupby(list(group_cols))[available].mean().reset_index()

    # Rank each metric (ascending rank = better)
    for metric in available:
        higher_better = metrics[metric]
        agg[f"{metric}_rank"] = agg[metric].rank(ascending=not higher_better, method="min")

    # Weighted composite rank (lower = better)
    rank_cols = [f"{m}_rank" for m in available]
    w_arr = np.array([weights.get(m, 1.0) for m in available])
    w_arr = w_arr / w_arr.sum()  # normalize

    agg["composite_rank"] = sum(
        agg[f"{m}_rank"] * weights.get(m, 1.0) for m in available
    ) / sum(weights.get(m, 1.0) for m in available)

    # Composite score: invert rank so higher = better
    max_rank = agg["composite_rank"].max()
    agg["composite_score"] = max_rank - agg["composite_rank"] + 1

    agg = agg.sort_values("composite_rank").reset_index(drop=True)
    return agg


def write_composite_ranking(
    matrix_csv: Path,
    output_dir: Path,
    *,
    metrics: dict[str, bool] | None = None,
    weights: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Read matrix CSV, compute ranking, write outputs.

    Outputs:
    - ``composite_ranking.csv``
    - ``composite_ranking.json`` (top-1 method for S10)
    """
    df = pd.read_csv(matrix_csv)
    ranking = compute_composite_ranking(df, metrics=metrics, weights=weights)

    ranking.to_csv(output_dir / "composite_ranking.csv", index=False)

    # Export top-1 for downstream
    top1 = ranking.iloc[0]
    top1_info = {
        "top1_method": str(top1.get("model", "")),
        "method": str(top1.get("model", "")),
        "model": str(top1.get("model", "")),
        "condition": str(top1.get("condition", "")),
        "composite_rank": float(top1.get("composite_rank", 0)),
        "composite_score": float(top1.get("composite_score", 0)),
        "metrics": {m: float(top1.get(m, 0)) for m in (metrics or METRIC_DIRECTIONS) if m in ranking.columns},
    }
    (output_dir / "top1_method.json").write_text(
        json.dumps(top1_info, indent=2), encoding="utf-8"
    )

    logger.info(f"Composite ranking written to {output_dir / 'composite_ranking.csv'}")
    logger.info(f"Top-1 method: {top1_info['model']} ({top1_info['condition']})")

    return ranking
