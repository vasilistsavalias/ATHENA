import unittest
from unittest.mock import MagicMock, patch
import numpy as np
import shutil
from pathlib import Path

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

# Mock statsmodels before importing the pipeline
mock_statsmodels = MagicMock()
sys.modules["statsmodels"] = mock_statsmodels
sys.modules["statsmodels.stats"] = mock_statsmodels.stats
sys.modules["statsmodels.stats.power"] = mock_statsmodels.stats.power

from thesis_pipeline.analysis.statistical_rigor import StatisticalTestingPipeline

class TestStatisticalRigor(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path("tests/temp_stats_rigor")
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
        self.test_dir.mkdir(parents=True, exist_ok=True)
        self.pipeline = StatisticalTestingPipeline(self.test_dir)

    def tearDown(self):
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    def test_power_analysis(self):
        # Setup mock return
        mock_analysis = mock_statsmodels.stats.power.TTestIndPower.return_value
        mock_analysis.solve_power.return_value = 64.0
        
        report = self.pipeline.run_power_analysis(effect_size=0.5)
        self.assertEqual(report["required_sample_size"], 64)
        self.assertTrue((self.test_dir / "power_analysis.json").exists())

    def test_normality(self):
        data = np.random.normal(0, 1, 100)
        res = self.pipeline.check_normality(data, "test_dist")
        # Since it's random, we don't assert truthiness of is_normal, 
        # but that the file was written and result is a dict.
        self.assertIsInstance(res, dict)
        self.assertTrue((self.test_dir / "normality_test_dist.json").exists())

if __name__ == "__main__":
    unittest.main()