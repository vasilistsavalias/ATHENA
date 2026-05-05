import os
import pytest
import sys

if os.environ.get("TORCH_IMPORT_OK", "1") != "1":
    pytest.skip(
        "Skipping torch-dependent tests because torch import failed/crashed in a subprocess probe. "
        "Fix your torch install or set TORCH_IMPORT_OK=1 to force-run.",
        allow_module_level=True,
    )

pytest.importorskip("torch")

import unittest
from unittest.mock import MagicMock, patch
from pathlib import Path
import sys
import os
import shutil
import numpy as np
from PIL import Image
from box import ConfigBox

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from thesis_pipeline.components.masking import MaskingStrategy
from thesis_pipeline.stages.stage_11_feature_engineering import FeatureEngineeringStage

class TestStage06(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path("tests/temp_stage06")
        self.input_dir = self.test_dir / "splits"
        self.output_dir = self.test_dir / "inpainting"
        self.report_dir = self.test_dir / "report"
        
        # Create Dummy Split Data
        for split in ['train', 'validation', 'test']:
            (self.input_dir / split).mkdir(parents=True, exist_ok=True)
            img = Image.new('RGB', (100, 100), color='red')
            img.save(self.input_dir / split / "test_img.png")

        # Mock ConfigManager
        self.mock_config_manager = MagicMock()
        self.mock_config_manager.get_feature_engineering_config.return_value = ConfigBox({
            "mask_strategy": "random_rectangle",
            "mask_config": ConfigBox({
                "min_mask_size_ratio": 0.1,
                "max_mask_size_ratio": 0.4
            })
        })
        self.mock_config_manager.get_paths.return_value = ConfigBox({
            "data": {
                "splits": str(self.input_dir),
                "inpainting": str(self.output_dir)
            },
            "artifacts": {
                "stage_10": str(self.report_dir)
            }
        })

    def tearDown(self):
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    def test_masking_strategy_random_rectangle(self):
        # Test Component Logic directly
        strategy = MaskingStrategy(
            "random_rectangle",
            {"min_mask_size_ratio": 0.1, "max_mask_size_ratio": 0.4},
            output_root=self.report_dir
        )
        mask = strategy.generator.generate(100, 100)
        self.assertEqual(mask.shape, (100, 100))
        # Mask should contain 0s and 255s
        self.assertTrue(np.any(mask == 255))
        self.assertTrue(np.any(mask == 0))

    @patch('thesis_pipeline.components.masking.ThesisPlotter')
    def test_stage_execution(self, mock_plotter):
        # Test Full Stage Execution
        stage = FeatureEngineeringStage(self.mock_config_manager)
        stage.run()
        
        # Check Output Structure
        for split in ['train', 'validation', 'test']:
            self.assertTrue((self.output_dir / split / "ground_truth").exists())
            self.assertTrue((self.output_dir / split / "masks").exists())
            self.assertTrue((self.output_dir / split / "ground_truth/test_img.png").exists())
            self.assertTrue((self.output_dir / split / "masks/test_img.png").exists())

    @patch('thesis_pipeline.components.masking.ThesisPlotter')
    def test_stage_execution_multi_mask_types(self, mock_plotter):
        self.mock_config_manager.get_feature_engineering_config.return_value = ConfigBox({
            "mask_types": ["irregular", "rect", "edge"],
            "mask_config": ConfigBox({
                "min_mask_size_ratio": 0.1,
                "max_mask_size_ratio": 0.4
            })
        })

        stage = FeatureEngineeringStage(self.mock_config_manager)
        stage.run()

        for split in ['train', 'validation', 'test']:
            self.assertTrue((self.output_dir / split / "masks/test_img.png").exists())

if __name__ == '__main__':
    unittest.main()


