"""
V7 — Deep baseline fine-tuning adapters.

Each adapter wraps a specific deep inpainting model (LaMa, MAT, CoModGAN)
with a uniform interface for domain-specific fine-tuning on the project's
mask/image pairs.
"""
from thesis_pipeline.components.baseline_finetuning.adapter import (
    BaselineAdapter,
    AdapterRegistry,
)

__all__ = ["BaselineAdapter", "AdapterRegistry"]
