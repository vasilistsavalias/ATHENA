import unittest
import shutil
from pathlib import Path
import time
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from thesis_pipeline.components.splitting import DataSplitter

class TestSplitting(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path("tests/temp_data_split")
        self.input_dir = self.test_dir / "input"
        self.output_dir = self.test_dir / "output"
        
        self.input_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create 10 dummy files
        for i in range(10):
            (self.input_dir / f"file_{i}.png").touch()

    def tearDown(self):
        if self.test_dir.exists():
            # On Windows, file handles (especially CSV writes) can linger briefly.
            # Retry a few times to avoid flaky teardown failures.
            for _ in range(8):
                try:
                    shutil.rmtree(self.test_dir)
                    return
                except PermissionError:
                    time.sleep(0.2)
            shutil.rmtree(self.test_dir)

    def test_split_data(self):
        splitter = DataSplitter(
            input_dir=self.input_dir,
            output_dir=self.output_dir,
            report_dir=self.output_dir,
            test_size=0.2, # 2 files
            validation_size=0.2, # 2 files
            random_state=42
        )
        splitter.split_data()
        
        # Check counts
        train_files = list((self.output_dir / "train").glob("*"))
        val_files = list((self.output_dir / "validation").glob("*"))
        test_files = list((self.output_dir / "test").glob("*"))
        
        self.assertEqual(len(train_files), 6)
        self.assertEqual(len(val_files), 2)
        self.assertEqual(len(test_files), 2)

if __name__ == '__main__':
    unittest.main()
