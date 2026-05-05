"""Tests for ThesisMetrics — specifically the Lab-range colour fidelity fix
and the NaN fallback for unavailable metrics.
"""
import math
import unittest
import numpy as np
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

from thesis_pipeline.components.evaluation.metrics import ThesisMetrics


class TestColorFidelity(unittest.TestCase):
    """Verify calculate_color_fidelity returns sensible values."""

    def setUp(self):
        # Initialise with device=cpu so we don't need CUDA
        self.metrics = ThesisMetrics(device="cpu")

    def test_identical_images_score_one(self):
        img = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        score = self.metrics.calculate_color_fidelity(img, img)
        self.assertAlmostEqual(score, 1.0, places=5,
                               msg="Identical images should have fidelity = 1.0")

    def test_different_images_score_less_than_one(self):
        # Use images with varied pixel values so Lab histograms differ meaningfully
        rng = np.random.RandomState(0)
        a = rng.randint(0, 128, (64, 64, 3), dtype=np.uint8)   # dark
        b = rng.randint(128, 256, (64, 64, 3), dtype=np.uint8)  # bright
        score = self.metrics.calculate_color_fidelity(a, b)
        self.assertLess(score, 1.0)
        self.assertGreater(score, 0.0)

    def test_score_in_zero_one_range(self):
        """Fidelity must always be in (0, 1]."""
        for _ in range(5):
            a = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
            b = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
            score = self.metrics.calculate_color_fidelity(a, b)
            self.assertGreater(score, 0.0)
            self.assertLessEqual(score, 1.0)


class TestNaNFallbacks(unittest.TestCase):
    """Verify that unavailable metrics return NaN, not 0.0."""

    def test_clip_returns_nan_when_unavailable(self):
        m = ThesisMetrics.__new__(ThesisMetrics)
        m.use_clip = False
        m.logger = __import__("logging").getLogger("test")
        from PIL import Image
        img = Image.new("RGB", (64, 64))
        result = m.calculate_clip_score(img, "some text")
        self.assertTrue(math.isnan(result), "CLIP should return NaN when unavailable")


class TestPatternPreservation(unittest.TestCase):
    def setUp(self):
        self.metrics = ThesisMetrics(device="cpu")

    def test_identical_images_high_score(self):
        img = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        score = self.metrics.calculate_pattern_preservation(img, img)
        self.assertGreater(score, 0.8,
                           msg="Identical images should have high pattern preservation")


if __name__ == "__main__":
    unittest.main()
