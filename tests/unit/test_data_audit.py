import unittest
import pandas as pd
import os
import json
from pathlib import Path
import shutil
import os
import sys
import pytest

if os.environ.get("TORCH_IMPORT_OK", "1") != "1":
    pytest.skip(
        "Skipping torch-dependent tests because torch import failed/crashed in a subprocess probe. "
        "Fix your torch install or set TORCH_IMPORT_OK=1 to force-run.",
        allow_module_level=True,
    )

pytest.importorskip("torch")
from thesis_pipeline.analysis.bias_analyzer import BiasAnalyzer
from thesis_pipeline.analysis.caption_analysis import CaptionQualityAnalyzer

class TestDataAuditAnalyzers(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path("tests/temp_audit_analysis")
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
        self.test_dir.mkdir(parents=True, exist_ok=True)
        
        # Create dummy rejection log
        self.log_file = self.test_dir / "rejection_log.csv"
        df = pd.DataFrame([
            {"filename": "img1.jpg", "yolo_confidence": 0.9, "blur_variance": 150.0, "reason": "Pass", "status": "Accepted"},
            {"filename": "img2.jpg", "yolo_confidence": 0.2, "blur_variance": "N/A", "reason": "No Object", "status": "Rejected"},
            {"filename": "img3.jpg", "yolo_confidence": 0.8, "blur_variance": 5.0, "reason": "Blurry", "status": "Rejected"}
        ])
        df.to_csv(self.log_file, index=False)
        
        # Create dummy captions
        self.captions_dir = self.test_dir / "captions"
        self.captions_dir.mkdir()
        (self.captions_dir / "img1.txt").write_text("A beautiful Greek amphora with black figures.", encoding='utf-8')
        (self.captions_dir / "img2.txt").write_text("Geometric pottery decoration from ancient era.", encoding='utf-8')

    def tearDown(self):
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    def test_bias_analyzer(self):
        analyzer = BiasAnalyzer(self.log_file, self.test_dir / "bias_reports")
        analyzer.analyze()
        self.assertTrue((self.test_dir / "bias_reports/rejection_reasons.png").exists())
        self.assertTrue((self.test_dir / "bias_reports/blur_bias_analysis.png").exists())

    def test_caption_analyzer(self):
        analyzer = CaptionQualityAnalyzer(self.captions_dir, self.test_dir / "caption_reports")
        analyzer.analyze()
        vocab_file = self.test_dir / "caption_reports/caption_vocabulary.json"
        self.assertTrue(vocab_file.exists())
        with open(vocab_file, 'r') as f:
            data = json.load(f)
            self.assertIn("key_term_frequency", data)
            self.assertGreater(data["total_words"], 0)

if __name__ == "__main__":
    unittest.main()
