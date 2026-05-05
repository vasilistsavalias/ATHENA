import unittest
import numpy as np
import cv2
from pathlib import Path
import shutil
import matplotlib
matplotlib.use('Agg')
from thesis_pipeline.components.masking import MaskingStrategy
from thesis_pipeline.analysis.realism_validator import RealismValidator

class TestSyntheticDamageRigor(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path("tests/temp_damage_rigor")
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
        self.test_dir.mkdir(parents=True, exist_ok=True)
        
        # Create dummy source images
        self.src_dir = self.test_dir / "src"
        self.src_dir.mkdir()
        for i in range(5):
            img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            cv2.imwrite(str(self.src_dir / f"img{i}.png"), img)

    def tearDown(self):
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    def test_irregular_mask_creation(self):
        # Use relaxed coverage settings for small test images (100x100)
        config = {"min_coverage": 0.05, "max_coverage": 0.95, "foreground_roi": False, "max_retries": 5}
        strategy = MaskingStrategy("irregular", config, output_root=self.test_dir)
        strategy.create_inpainting_dataset(self.src_dir, self.test_dir / "output")
        
        mask_dir = self.test_dir / "output/masks"
        self.assertTrue(mask_dir.exists())
        self.assertEqual(len(list(mask_dir.glob("*.png"))), 5)

    def test_realism_validator(self):
        # Create some dummy masks (one box, one irregular)
        mask_dir = self.test_dir / "masks"
        mask_dir.mkdir()
        
        # Box
        box = np.zeros((100, 100), dtype=np.uint8)
        box[10:90, 10:90] = 255
        cv2.imwrite(str(mask_dir / "box.png"), box)
        
        # Irregular (L-shape)
        irr = np.zeros((100, 100), dtype=np.uint8)
        irr[10:90, 10:20] = 255
        irr[80:90, 10:90] = 255
        cv2.imwrite(str(mask_dir / "irr.png"), irr)
        
        validator = RealismValidator(self.test_dir / "reports")
        df = validator.analyze_mask_geometry(mask_dir, "test")
        
        self.assertIsNotNone(df)
        # Irregular should have higher complexity than box
        box_stats = df[df['filename'] == "box.png"].iloc[0]
        irr_stats = df[df['filename'] == "irr.png"].iloc[0]
        self.assertGreater(irr_stats['complexity'], box_stats['complexity'])

if __name__ == "__main__":
    unittest.main()
