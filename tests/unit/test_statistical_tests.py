import unittest
import pandas as pd
import numpy as np
from pathlib import Path
import shutil
from thesis_pipeline.analysis.statistical_tests import StatisticalComparator

class TestStatisticalComparator(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path("tests/temp_stats_test")
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
        self.test_dir.mkdir(parents=True, exist_ok=True)
        
        # Create dummy data
        self.model_df = pd.DataFrame([
            {"filename": f"img{i}.jpg", "psnr": 25.0 + np.random.randn()} for i in range(20)
        ])
        self.baseline_df = pd.DataFrame([
            {"filename": f"img{i}.jpg", "psnr": 20.0 + np.random.randn()} for i in range(20)
        ])

    def tearDown(self):
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    def test_comparison(self):
        comparator = StatisticalComparator(self.test_dir)
        report = comparator.compare(self.model_df, self.baseline_df, "OurModel", "Baseline")
        
        self.assertIsNotNone(report)
        self.assertTrue((self.test_dir / "comparison_OurModel_vs_Baseline_psnr.json").exists())
        self.assertTrue((self.test_dir / "comparison_OurModel_vs_Baseline_psnr.png").exists())
        self.assertTrue(report["is_significant"]) # 25 vs 20 should be significant

if __name__ == "__main__":
    unittest.main()
