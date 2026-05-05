import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.power import TTestIndPower
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class StatisticalTestingPipeline:
    """
    Handles advanced statistical rigor: power analysis, normality checks, and significance testing.
    """
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run_power_analysis(self, effect_size: float = 0.5, alpha: float = 0.05, power: float = 0.8):
        """
        Determines the required sample size to detect a given effect size.
        """
        analysis = TTestIndPower()
        sample_size = analysis.solve_power(effect_size=effect_size, power=power, alpha=alpha, ratio=1.0)
        
        report = {
            "required_sample_size": int(np.ceil(sample_size)),
            "alpha": alpha,
            "target_power": power,
            "effect_size_target": effect_size
        }
        
        import json
        with open(self.output_dir / "power_analysis.json", 'w') as f:
            json.dump(report, f, indent=4)
        
        logger.info(f"Power Analysis: Required N={report['required_sample_size']}")
        return report

    def check_normality(self, data: np.ndarray, name: str):
        """Runs Shapiro-Wilk and saves result."""
        stat, p = stats.shapiro(data)
        res = {
            "test": "Shapiro-Wilk",
            "stat": float(stat),
            "p_value": float(p),
            "is_normal": bool(p > 0.05)
        }
        with open(self.output_dir / f"normality_{name}.json", 'w') as f:
            import json
            json.dump(res, f, indent=4)
        return res
