import os
import random
import numpy as np
import torch
import logging

logger = logging.getLogger(__name__)

class SeedManager:
    """
    Centralized seed management to ensure determinism across all libraries.
    Sets seeds for: Python stdlib, NumPy, PyTorch (CPU+CUDA), CuDNN,
    PYTHONHASHSEED, torch deterministic algorithms, and HuggingFace transformers.
    """
    @staticmethod
    def set_seed(seed: int):
        """Sets the seed for all libraries to ensure full reproducibility."""
        logger.info(f"Setting global seed: {seed}")
        
        # 1. Python hash seed (must be set before any hashing occurs ideally,
        #    but setting env var still documents intent and affects subprocesses)
        os.environ["PYTHONHASHSEED"] = str(seed)
        
        # 2. Python stdlib
        random.seed(seed)
        
        # 3. NumPy
        np.random.seed(seed)
        
        # 4. PyTorch CPU + all CUDA devices
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
        # 5. CuDNN deterministic behavior
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        # 6. PyTorch deterministic algorithms (errors on non-deterministic ops)
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            # Some environments / PyTorch builds don't support this fully
            logger.debug("torch.use_deterministic_algorithms(True) not available; skipping.")
        
        # 7. HuggingFace transformers (sets internal seeds for generate, dropout, etc.)
        try:
            from transformers import set_seed as hf_set_seed
            hf_set_seed(seed)
        except ImportError:
            logger.debug("transformers not installed; skipping HF seed.")
        
        logger.info(
            f"Seeds set: PYTHONHASHSEED={seed}, random/numpy/torch/cuda/cudnn/deterministic_algorithms"
        )
        
    @staticmethod
    def get_seed_from_config(config_manager):
        """Extracts seed from config or returns default 42."""
        return config_manager.config.get("global_params", {}).get("random_state", 42)
