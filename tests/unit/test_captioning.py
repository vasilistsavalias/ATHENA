import unittest
import shutil
from pathlib import Path
from PIL import Image
from unittest.mock import MagicMock, patch
import sys
import os
import pytest
import torch

if os.environ.get("TORCH_IMPORT_OK", "1") != "1":
    pytest.skip(
        "Skipping torch-dependent tests because torch import failed/crashed in a subprocess probe. "
        "Fix your torch install or set TORCH_IMPORT_OK=1 to force-run.",
        allow_module_level=True,
    )

pytest.importorskip("torch")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from thesis_pipeline.components.captioning.local_vlm import LocalVLM
from thesis_pipeline.stages.stage_07_caption_generation import CaptionGenerationStage

class TestCaptioning(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path("tests/temp_data_cap")
        self.test_dir.mkdir(parents=True, exist_ok=True)
        
        # Create test image
        self.img_path = self.test_dir / "test.jpg"
        Image.new('RGB', (100, 100), color='red').save(self.img_path)

    def tearDown(self):
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    @patch('thesis_pipeline.components.captioning.local_vlm.Blip2Processor')
    @patch('thesis_pipeline.components.captioning.local_vlm.Blip2ForConditionalGeneration')
    def test_caption_generation_mocked(self, mock_model_cls, mock_processor_cls):
        # Setup Mocks
        mock_model = mock_model_cls.from_pretrained.return_value
        mock_processor = mock_processor_cls.from_pretrained.return_value
        mock_model.to.return_value = mock_model
        mock_model.eval.return_value = None
        
        # Mock generate method
        mock_model.generate.return_value = torch.tensor([[1, 2, 3]])  # Mock token IDs
        
        # Mock Decode
        mock_processor.batch_decode.return_value = ["Mocked Caption"]
        
        # Mock processor inputs
        mock_inputs = {
            "pixel_values": torch.zeros((1, 3, 2, 2), dtype=torch.float32),
            "input_ids": torch.tensor([[1, 2]]),
            "attention_mask": torch.tensor([[1, 1]]),
        }
        mock_processor.return_value = mock_inputs
        
        # Initialize LocalVLM with mocks
        print("\nRunning Mocked BLIP-2 Test with LocalVLM...")
        vlm = LocalVLM()
        
        # Generate caption for test image
        caption = vlm.generate_caption(str(self.img_path))
        
        # Verify
        self.assertIsNotNone(caption)
        print(f"Generated Caption: {caption}")

    def test_local_vlm_retries_once_after_oom(self):
        image = Image.new("RGB", (32, 32), color="red")
        vlm = LocalVLM.__new__(LocalVLM)
        vlm.device = "cuda:0"
        vlm.logger = MagicMock()

        with patch.object(
            vlm,
            "_generate_once",
            side_effect=[torch.cuda.OutOfMemoryError("oom"), "Recovered caption"],
        ), patch.object(vlm, "_cleanup_device_memory", return_value=None):
            report = vlm._generate_caption_with_report(
                image,
                str(self.img_path),
                "describe",
                prompt_label="description",
                max_new_tokens=50,
                oom_retry_limit=1,
                oom_backoff_max_new_tokens=24,
            )

        self.assertEqual(report["text"], "Recovered caption")
        self.assertEqual(report["retry_count"], 1)
        self.assertEqual(report["attempts"], 2)

    def test_caption_generation_failure_policy_raises(self):
        stage = CaptionGenerationStage.__new__(CaptionGenerationStage)
        stage.output_dir = self.test_dir
        stage.caption_cfg = {
            "failure_policy": "hard_fail",
            "success_rate_min": 0.9,
            "max_worker_fail_streak": 3,
        }

        report = {
            "success_rate": 0.5,
            "model_load_errors": [],
            "max_fail_streak_observed": 1,
        }

        with self.assertRaises(RuntimeError):
            stage._enforce_failure_policy(report)
        self.assertTrue((self.test_dir / "caption_generation_report.json").exists())

if __name__ == '__main__':
    unittest.main()


