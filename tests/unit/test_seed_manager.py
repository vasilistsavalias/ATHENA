import os
import unittest
import numpy as np
import random
import pytest
import sys

if os.environ.get("TORCH_IMPORT_OK", "1") != "1":
    pytest.skip(
        "Skipping torch-dependent tests because torch import failed/crashed in a subprocess probe. "
        "Fix your torch install or set TORCH_IMPORT_OK=1 to force-run.",
        allow_module_level=True,
    )

torch = pytest.importorskip("torch")
from thesis_pipeline.utils.seed_manager import SeedManager

class TestSeedManager(unittest.TestCase):
    def test_determinism(self):
        """Verify that setting the seed produces identical random numbers."""
        seed = 42
        
        # Run 1
        SeedManager.set_seed(seed)
        py_rand1 = random.random()
        np_rand1 = np.random.rand(1)
        torch_rand1 = torch.rand(1)
        
        # Run 2
        SeedManager.set_seed(seed)
        py_rand2 = random.random()
        np_rand2 = np.random.rand(1)
        torch_rand2 = torch.rand(1)
        
        self.assertEqual(py_rand1, py_rand2)
        self.assertEqual(np_rand1, np_rand2)
        self.assertTrue(torch.equal(torch_rand1, torch_rand2))

    def test_pythonhashseed_is_set(self):
        """PYTHONHASHSEED env var must be set to the seed value."""
        SeedManager.set_seed(123)
        self.assertEqual(os.environ.get("PYTHONHASHSEED"), "123")

    def test_deterministic_algorithms_enabled(self):
        """torch.use_deterministic_algorithms should be True after set_seed."""
        SeedManager.set_seed(42)
        self.assertTrue(torch.are_deterministic_algorithms_enabled())

if __name__ == "__main__":
    unittest.main()
