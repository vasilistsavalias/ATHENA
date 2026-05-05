import shutil
import unittest
from pathlib import Path
from types import SimpleNamespace

from PIL import Image

from thesis_pipeline.stages.stage_11_feature_engineering import MaskRealismStage


class _DummyConfig(dict):
    """Minimal config that supports both attribute and dict access."""
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)


class _DummyConfigManager:
    def __init__(self):
        self.config = _DummyConfig()

    def get_feature_engineering_config(self):
        return {}

    def get_paths(self):
        return SimpleNamespace(
            artifacts=SimpleNamespace(
                root="tests/temp_mask_realism/artifacts",
                stage_10="tests/temp_mask_realism/artifacts/10_feature_engineering",
                stage_11="tests/temp_mask_realism/artifacts/11_feature_engineering",
            ),
            data=SimpleNamespace(inpainting="tests/temp_mask_realism/inpainting"),
        )


class TestMaskRealismStage(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path("tests/temp_mask_realism")
        self.test_dir.mkdir(parents=True, exist_ok=True)

        self.m1 = self.test_dir / "m1.png"
        self.m2 = self.test_dir / "m2.png"

        Image.new("L", (64, 64), color=0).save(self.m1)
        Image.new("L", (128, 96), color=0).save(self.m2)

    def tearDown(self):
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    def test_compute_avg_magnitude_spectrum_handles_mixed_shapes(self):
        stage = MaskRealismStage(_DummyConfigManager())
        avg = stage._compute_avg_magnitude_spectrum([self.m1, self.m2])

        self.assertIsNotNone(avg)
        self.assertEqual(avg.shape, stage._ANALYSIS_TARGET_SIZE[::-1])  # numpy is (H, W)


if __name__ == "__main__":
    unittest.main()


