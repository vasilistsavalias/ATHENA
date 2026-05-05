import unittest
import shutil
from pathlib import Path
from PIL import Image
import pandas as pd
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from thesis_pipeline.components.exploratory_data_analysis import ExploratoryDataAnalyzer

class TestEDA(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path("tests/temp_data")
        self.input_dir = self.test_dir / "input"
        self.output_dir = self.test_dir / "output"
        
        self.input_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create dummy images
        Image.new('RGB', (100, 100)).save(self.input_dir / "test1.jpg")
        Image.new('RGB', (200, 200)).save(self.input_dir / "test2.png")

    def tearDown(self):
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    def test_eda_run(self):
        analyzer = ExploratoryDataAnalyzer(
            input_dir=self.input_dir,
            output_dir=self.output_dir,
            extensions=[".jpg", ".png"]
        )
        analyzer.run()
        
        # Verify CSV
        csv_path = self.output_dir / "full_image_metadata.csv"
        self.assertTrue(csv_path.exists())
        df = pd.read_csv(csv_path)
        self.assertEqual(len(df), 2)
        
        # Verify Plots
        self.assertTrue((self.output_dir / "width_distribution.png").exists())

if __name__ == '__main__':
    unittest.main()
