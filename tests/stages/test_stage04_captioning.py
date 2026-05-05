import unittest
import shutil
from pathlib import Path
from PIL import Image
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from thesis_pipeline.components.processing import ImageProcessor

class TestProcessing(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path("tests/temp_data_proc")
        self.input_dir = self.test_dir / "input"
        self.output_dir = self.test_dir / "output"
        
        self.input_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create dummy image (rectangular)
        Image.new('RGB', (100, 200)).save(self.input_dir / "test.jpg")

    def tearDown(self):
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    def test_process_images(self):
        processor = ImageProcessor(
            input_dir=self.input_dir,
            output_dir=self.output_dir,
            report_dir=self.output_dir, # Use output dir for reports in tests
            image_size=(512, 512),
            output_format="PNG"
        )
        processor.process_images()
        
        output_file = self.output_dir / "test.png"
        self.assertTrue(output_file.exists())
        
        with Image.open(output_file) as img:
            self.assertEqual(img.size, (512, 512))
            self.assertEqual(img.format, "PNG")

if __name__ == '__main__':
    unittest.main()
