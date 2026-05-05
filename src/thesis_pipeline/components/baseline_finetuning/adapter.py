"""
Baseline adapter interface and registry for deep inpainting model fine-tuning.
"""
from __future__ import annotations

import abc
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Type

logger = logging.getLogger(__name__)


@dataclass
class FTResult:
    """Summary returned after fine-tuning a single baseline model."""

    model_name: str
    best_epoch: int
    best_val_loss: float
    total_epochs: int
    checkpoint_path: str
    metrics: dict = field(default_factory=dict)


class BaselineAdapter(abc.ABC):
    """Uniform interface for fine-tuning a deep inpainting baseline.

    Subclasses must implement:
    - ``setup()`` — download/prepare model & training scaffold
    - ``train()`` — run fine-tuning loop
    - ``export()`` — export weights for inference via ``DeepBaselineRunner``
    """

    def __init__(
        self,
        model_name: str,
        cfg: dict[str, Any],
        data_root: Path,
        checkpoint_dir: Path,
        device: str = "cuda",
    ):
        self.model_name = model_name
        self.cfg = cfg
        self.data_root = data_root
        self.checkpoint_dir = checkpoint_dir / model_name
        self.device = device
        self.logger = logging.getLogger(f"{__name__}.{model_name}")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Data layout helpers — assume S04 produced the standard structure
    # ------------------------------------------------------------------

    @property
    def train_images(self) -> Path:
        return self.data_root / "train" / "ground_truth"

    @property
    def train_masks(self) -> Path:
        return self.data_root / "train" / "masks"

    @property
    def val_images(self) -> Path:
        return self.data_root / "validation" / "ground_truth"

    @property
    def val_masks(self) -> Path:
        return self.data_root / "validation" / "masks"

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def setup(self) -> None:
        """Download model weights, compile training code if needed."""

    @abc.abstractmethod
    def train(
        self,
        epochs: int,
        batch_size: int,
        learning_rate: float,
        early_stopping_patience: int = 10,
    ) -> FTResult:
        """Run the fine-tuning loop. Return a ``FTResult``."""

    @abc.abstractmethod
    def export(self) -> Path:
        """Export the best checkpoint to a location ``DeepBaselineRunner`` can load."""


# ---------------------------------------------------------------------------
# Adapter Registry
# ---------------------------------------------------------------------------

class AdapterRegistry:
    """Map model names → adapter classes.  Lazy-imports so we don't require
    all dependencies to be installed when only a subset is used."""

    _REGISTRY: Dict[str, str] = {
        "LaMa": "thesis_pipeline.components.baseline_finetuning.lama_adapter.LamaAdapter",
        "MAT": "thesis_pipeline.components.baseline_finetuning.mat_adapter.MATAdapter",
        "CoModGAN": "thesis_pipeline.components.baseline_finetuning.comodgan_adapter.CoModGANAdapter",
    }

    @classmethod
    def available_models(cls) -> list[str]:
        return list(cls._REGISTRY.keys())

    @classmethod
    def get(cls, name: str) -> Type[BaselineAdapter]:
        """Return the adapter class for *name*.  Raises ``ValueError`` for unknown names."""
        fqn = cls._REGISTRY.get(name)
        if fqn is None:
            raise ValueError(f"Unknown baseline adapter: {name!r}. Available: {list(cls._REGISTRY)}")
        module_path, class_name = fqn.rsplit(".", 1)
        import importlib

        mod = importlib.import_module(module_path)
        return getattr(mod, class_name)
