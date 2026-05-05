import unittest
import os
import csv
from pathlib import Path
import pytest
import sys

if os.environ.get("TORCH_IMPORT_OK", "1") != "1":
    pytest.skip(
        "Skipping torch-dependent tests because torch import failed/crashed in a subprocess probe. "
        "Fix your torch install or set TORCH_IMPORT_OK=1 to force-run.",
        allow_module_level=True,
    )

pytest.importorskip("torch")
from thesis_pipeline.utils.resource_tracker import ResourceTracker

class TestResourceTracker(unittest.TestCase):
    def setUp(self):
        self.log_file = Path("tests/temp/resource_log.csv")
        if self.log_file.exists():
            os.remove(self.log_file)
        self.tracker = ResourceTracker(self.log_file)

    def tearDown(self):
        if self.log_file.exists():
            os.remove(self.log_file)
        if self.log_file.parent.exists():
            self.log_file.parent.rmdir()

    def test_tracking(self):
        @self.tracker.track("TestStage")
        def dummy_func():
            import time
            time.sleep(0.1)
            return True

        result = dummy_func()
        self.assertTrue(result)
        self.assertTrue(self.log_file.exists())
        
        with open(self.log_file, 'r') as f:
            reader = list(csv.reader(f))
            self.assertEqual(len(reader), 2) # Header + 1 entry
            self.assertEqual(reader[1][1], "TestStage")
            self.assertGreater(float(reader[1][2]), 0.05)

if __name__ == "__main__":
    unittest.main()
