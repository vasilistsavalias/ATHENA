import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class StatisticalComparator:
    """
    Compares two model results (e.g., Our Model vs Baseline) for statistical significance.
    """
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def compare(self, model_df: pd.DataFrame, baseline_df: pd.DataFrame, model_name: str, baseline_name: str, metric: str = 'psnr'):
        """
        Runs paired tests between model and baseline on a specific metric.
        """
        # Ensure they are aligned by filename
        merged = pd.merge(
            model_df[['filename', metric]], 
            baseline_df[['filename', metric]], 
            on='filename', 
            suffixes=(f'_{model_name}', f'_{baseline_name}')
        )
        
        if merged.empty:
            logger.error("No common files found between results for comparison.")
            return

        col_m = f"{metric}_{model_name}"
        col_b = f"{metric}_{baseline_name}"
        
        data_m = merged[col_m].values
        data_b = merged[col_b].values

        # 1. Normality Test (Shapiro-Wilk)
        diff = data_m - data_b
        _, p_norm = stats.shapiro(diff)
        
        # 2. Significance Test
        if p_norm > 0.05:
            # Normal distribution -> Paired T-test
            t_stat, p_val = stats.ttest_rel(data_m, data_b)
            test_type = "Paired T-test"
        else:
            # Non-normal -> Wilcoxon Signed-Rank
            t_stat, p_val = stats.wilcoxon(data_m, data_b)
            test_type = "Wilcoxon Signed-Rank"

        # 3. Effect Size (Cohen's d)
        cohen_d = np.mean(diff) / np.std(diff, ddof=1)

        # 4. Save results
        report = {
            "metric": metric,
            "test_type": test_type,
            "p_value": float(p_val),
            "t_statistic": float(t_stat),
            "cohen_d": float(cohen_d),
            "mean_diff": float(np.mean(diff)),
            "is_significant": bool(p_val < 0.05)
        }
        
        report_file = self.output_dir / f"comparison_{model_name}_vs_{baseline_name}_{metric}.json"
        import json
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=4)

        # 5. Visual Comparison
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=merged[[col_m, col_b]])
        plt.title(f"{metric.upper()} Comparison: {model_name} vs {baseline_name}")
        plt.ylabel(metric.upper())
        plt.tight_layout()
        plt.savefig(self.output_dir / f"comparison_{model_name}_vs_{baseline_name}_{metric}.png")
        plt.close()

        logger.info(f"Statistical comparison complete. Sig: {p_val < 0.05}")
        return report
