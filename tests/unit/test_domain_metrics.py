import unittest
import numpy as np
import cv2
from thesis_pipeline.analysis.domain_metrics import DomainMetricCalculator

class TestDomainMetrics(unittest.TestCase):
    def setUp(self):
        self.calculator = DomainMetricCalculator()
        
        # 1. Create original with geometric pattern (diagonal lines)
        self.original = np.zeros((100, 100, 3), dtype=np.uint8)
        for i in range(0, 100, 10):
            cv2.line(self.original, (i, 0), (i+50, 100), (255, 255, 255), 2)
            
        # 2. Mask center
        self.mask = np.zeros((100, 100), dtype=np.uint8)
        self.mask[40:60, 40:60] = 255
        
        # 3. Perfect restoration
        self.restored_good = self.original.copy()
        
        # 4. Bad restoration (noise in mask area)
        self.restored_bad = self.original.copy()
        self.restored_bad[40:60, 40:60] = np.random.randint(0, 255, (20, 20, 3), dtype=np.uint8)

    def test_pattern_continuity(self):
        score_good = self.calculator.calculate_pattern_continuity(self.original, self.restored_good, self.mask)
        score_bad = self.calculator.calculate_pattern_continuity(self.original, self.restored_bad, self.mask)
        
        self.assertAlmostEqual(score_good, 1.0)
        self.assertLess(score_bad, score_good)

    def test_color_fidelity(self):
        score_good = self.calculator.calculate_color_fidelity(self.original, self.restored_good, self.mask)
        # Shift color in mask area
        res_shifted = self.original.copy()
        res_shifted[40:60, 40:60] = [255, 0, 0] # Blue shift
        score_bad = self.calculator.calculate_color_fidelity(self.original, res_shifted, self.mask)
        
        self.assertEqual(score_good, 1.0)
        self.assertLess(score_bad, score_good)

if __name__ == "__main__":
    unittest.main()
