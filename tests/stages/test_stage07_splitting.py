import unittest
import shutil
from pathlib import Path
from PIL import Image
import numpy as np
import sys
import os
from unittest.mock import MagicMock, patch
from box import ConfigBox

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from thesis_pipeline.components.masking import MaskingStrategy

class TestStage05(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path("tests/temp_data_stg5")
        self.input_dir = self.test_dir / "input"
        self.output_dir = self.test_dir / "output"
        
        self.input_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Dummy Image
        Image.new('RGB', (100, 100), color='white').save(self.input_dir / "test.png")

    def tearDown(self):
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    def test_masking_irregular(self):
        strategy = MaskingStrategy(
            strategy_name="irregular",
            mask_config={}
        )
        strategy.create_inpainting_dataset(self.input_dir, self.output_dir)
        
        mask_path = self.output_dir / "masks" / "test.png"
        self.assertTrue(mask_path.exists())
        
        # Verify mask is not empty (has some white pixels)
        with Image.open(mask_path) as img:
            arr = np.array(img)
            self.assertTrue(np.any(arr > 0))

if __name__ == '__main__':
    unittest.main()
