import shutil
import unittest
from pathlib import Path

from PIL import Image

from thesis_pipeline.components.processing import ImageProcessor


class TestImageProcessorLimit(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path("tests/temp_processing_limit")
        self.input_dir = self.test_dir / "input"
        self.output_dir = self.test_dir / "output"
        self.report_dir = self.test_dir / "report"

        self.input_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.report_dir.mkdir(parents=True, exist_ok=True)

        for i in range(5):
            img = Image.new("RGB", (16, 16), color=(i, i, i))
            img.save(self.input_dir / f"img_{i}.jpg")

    def tearDown(self):
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    def test_process_images_respects_max_images(self):
        processor = ImageProcessor(
            input_dir=self.input_dir,
            output_dir=self.output_dir,
            report_dir=self.report_dir,
            image_size=(8, 8),
            output_format="png",
        )
        processor.plotter.plot_image_grid = lambda *args, **kwargs: None

        summary = processor.process_images(max_images=3)

        self.assertEqual(summary["total_found"], 5)
        self.assertEqual(summary["selected_count"], 3)
        self.assertEqual(summary["processed_count"], 3)

        out_files = sorted(self.output_dir.glob("*.png"), key=lambda p: p.name)
        self.assertEqual(len(out_files), 3)
        self.assertTrue((self.output_dir / "img_0.png").exists())
        self.assertTrue((self.output_dir / "img_1.png").exists())
        self.assertTrue((self.output_dir / "img_2.png").exists())


if __name__ == "__main__":
    unittest.main()

