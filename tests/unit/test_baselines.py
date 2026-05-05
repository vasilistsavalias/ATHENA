import unittest
from unittest.mock import MagicMock, patch
from PIL import Image
import numpy as np
import shutil
from pathlib import Path
import sys
import os
import pytest

if os.environ.get("TORCH_IMPORT_OK", "1") != "1":
    pytest.skip(
        "Skipping torch-dependent tests because torch import failed/crashed in a subprocess probe. "
        "Fix your torch install or set TORCH_IMPORT_OK=1 to force-run.",
        allow_module_level=True,
    )

pytest.importorskip("torch")

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from thesis_pipeline.components.evaluation.baselines import TeleaInpainter, VanillaSDInpainter

class TestBaselines(unittest.TestCase):
    def test_telea_inpainting(self):
        """
        Test that OpenCV inpainting runs and returns an image of the same size.
        """
        inpainter = TeleaInpainter()
        
        # Create dummy data (White image with black hole)
        img = Image.new("RGB", (100, 100), color="white")
        mask = Image.new("L", (100, 100), color=0)
        
        # Draw a hole in mask (255) and image (black)
        mask_np = np.array(mask)
        mask_np[40:60, 40:60] = 255
        mask = Image.fromarray(mask_np)
        
        img_np = np.array(img)
        img_np[40:60, 40:60] = 0
        img = Image.fromarray(img_np)
        
        result = inpainter.inpaint(img, mask)
        
        self.assertEqual(result.size, (100, 100))
        self.assertIsInstance(result, Image.Image)
        
        # Check if it filled the hole (center pixel shouldn't be pure black 0)
        # Telea usually smooths it out.
        center_pixel = result.getpixel((50, 50))
        self.assertNotEqual(center_pixel, (0, 0, 0), "Telea failed to inpaint the black hole")

    @patch("thesis_pipeline.components.evaluation.baselines.StableDiffusionInpaintPipeline")
    def test_vanilla_sd_loading(self, mock_pipeline):
        """
        Test that SD loads the correct model ID and disables safety checker.
        """
        mock_pipe_instance = MagicMock()
        mock_pipeline.from_pretrained.return_value = mock_pipe_instance
        
        inpainter = VanillaSDInpainter(device="cpu")
        inpainter.load_model()
        
        # Verify model ID
        mock_pipeline.from_pretrained.assert_called_once()
        call_args = mock_pipeline.from_pretrained.call_args
        self.assertEqual(call_args[0][0], "runwayml/stable-diffusion-inpainting")
        
        # Verify Safety Checker was set to None
        self.assertIsNone(inpainter.pipeline.safety_checker)

if __name__ == '__main__':
    unittest.main()
