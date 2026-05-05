import unittest
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch
from box import ConfigBox
from PIL import Image
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
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from thesis_pipeline.components.model_evaluation import ModelEvaluator

class TestStage08(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path("tests/temp_eval")
        self.test_data_dir = self.test_dir / "data"
        self.output_dir = self.test_dir / "output"
        
        # Create directories
        (self.test_data_dir / "ground_truth").mkdir(parents=True, exist_ok=True)
        (self.test_data_dir / "masks").mkdir(parents=True, exist_ok=True)
        (self.test_data_dir / "captions").mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Dummy Data
        Image.new('RGB', (64, 64), color='red').save(self.test_data_dir / "ground_truth" / "test.png")
        Image.new('L', (64, 64), color=0).save(self.test_data_dir / "masks" / "test.png")
        (self.test_data_dir / "captions" / "test.txt").write_text("A red square", encoding="utf-8")
        
        self.config = ConfigBox({
            "trained_model_dir": "tests/temp_models",
            "output_dir": str(self.output_dir),
            "num_samples_to_evaluate": 1,
            "num_inference_steps": 1,
            "device": "cpu"
        })

    def tearDown(self):
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    @patch('thesis_pipeline.components.model_evaluation.StableDiffusionInpaintPipeline')
    @patch('thesis_pipeline.components.model_evaluation.UNet2DConditionModel')
    def test_evaluate(self, mock_unet, mock_pipeline_cls):
        # Mock Pipeline
        pipeline = mock_pipeline_cls.from_pretrained.return_value
        
        # Mock Inference Output
        output_obj = MagicMock()
        output_obj.images = [Image.new('RGB', (64, 64), color='blue')]
        pipeline.return_value = output_obj
        
        model_dir = Path("tests/temp_models")
        evaluator = ModelEvaluator(self.config, self.test_data_dir, output_dir=self.output_dir, model_dir=model_dir)
        evaluator.evaluate()
        
        # Verify output
        self.assertTrue((self.output_dir / "evaluation_metrics.csv").exists())
        self.assertTrue((self.output_dir / "samples" / "test" / "composite.png").exists())
        
        # Verify Caption Usage
        pipeline.assert_called()
        call_kwargs = pipeline.call_args[1]
        self.assertEqual(call_kwargs['prompt'], "A red square")

if __name__ == '__main__':
    unittest.main()
