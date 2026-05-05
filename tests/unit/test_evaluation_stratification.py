import unittest
import pandas as pd
import numpy as np
import shutil
from pathlib import Path
from thesis_pipeline.analysis.failure_analyzer import FailureAnalyzer

class TestEvaluationStratification(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path("tests/temp_strat_eval")
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
        self.test_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Create dummy metrics
        self.metrics_file = self.test_dir / "metrics.csv"
        df = pd.DataFrame([
            {"filename": f"img{i}.png", "psnr": 30.0 - i, "mask_coverage": 0.05 * i} for i in range(10)
        ])
        df.to_csv(self.metrics_file, index=False)
        
        # 2. Create dummy samples
        self.samples_dir = self.test_dir / "samples"
        for i in range(10):
            stem = f"img{i}"
            (self.samples_dir / stem).mkdir(parents=True)
            (self.samples_dir / stem / "composite.png").touch()

    def tearDown(self):
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    def test_failure_extraction(self):
        analyzer = FailureAnalyzer(self.metrics_file, self.test_dir / "failures")
        analyzer.extract_failures(self.samples_dir, top_n=3)
        
        failure_dir = self.test_dir / "failures"
        # img9, img8, img7 should be the worst (lowest PSNR)
        self.assertTrue((failure_dir / "img9").exists())
        self.assertTrue((failure_dir / "img8").exists())
        self.assertTrue((failure_dir / "failure_summary.csv").exists())

if __name__ == "__main__":
    unittest.main()
