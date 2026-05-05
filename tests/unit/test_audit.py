import unittest
from pathlib import Path
import csv
import shutil
from thesis_pipeline.components.audit import AuditLogger

class TestAuditLogger(unittest.TestCase):
    def setUp(self):
        self.output_dir = Path("tests/temp_audit")
        if self.output_dir.exists():
            shutil.rmtree(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        if self.output_dir.exists():
            shutil.rmtree(self.output_dir)

    def test_logging(self):
        logger = AuditLogger(self.output_dir)
        
        # Log some decisions
        logger.log_decision("img1.jpg", 0.95, 200.0, "Pass", "Accepted")
        logger.log_decision("img2.jpg", 0.30, None, "No Object", "Rejected")
        logger.log_decision("img3.jpg", 0.90, 5.0, "Blurry", "Rejected")
        
        log_file = self.output_dir / "rejection_log.csv"
        self.assertTrue(log_file.exists())
        
        with open(log_file, "r") as f:
            reader = list(csv.reader(f))
            self.assertEqual(len(reader), 4) # Header + 3 rows
            self.assertEqual(
                reader[0],
                [
                    "filename",
                    "reason",
                    "status",
                    "yolo_confidence",
                    "focus_metric",
                    "focus_score",
                    "focus_threshold_used",
                    "bbox_area_ratio",
                    "crop_index",
                    "blur_variance",
                ],
            )
            self.assertEqual(reader[2][1], "No Object")
            self.assertEqual(reader[2][2], "Rejected")

if __name__ == '__main__':
    unittest.main()
