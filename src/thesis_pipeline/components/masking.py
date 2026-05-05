# src/thesis_pipeline/components/masking.py
import logging
import shutil
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from thesis_pipeline.visualization import ThesisPlotter
from thesis_pipeline.components.mask_generators import BoxMaskGenerator, IrregularMaskGenerator, EdgeMaskGenerator


# ---------------------------------------------------------------------------
# Foreground ROI helpers
# ---------------------------------------------------------------------------

def _compute_foreground_mask(
    img_array: np.ndarray,
    *,
    bg_threshold: int = 20,
    min_area_frac: float = 0.10,
) -> np.ndarray:
    """Return a binary uint8 mask (255 = foreground) for *img_array*.

    Strategy
    --------
    1. Convert to grayscale.
    2. Threshold at ``bg_threshold`` — pixels darker than this on *every*
       channel are treated as background (typically black studio backdrop).
    3. Close small holes with a morphological closing pass.
    4. If the detected foreground is smaller than ``min_area_frac`` of the
       image, fall back to an all-foreground mask (the image has no obvious
       background).
    """
    if img_array.ndim == 3:
        # Max-channel thresholding: a pixel is background only if ALL
        # channels are below the threshold (dark on every channel).
        max_channel = img_array.max(axis=2)
    else:
        max_channel = img_array

    fg = (max_channel >= bg_threshold).astype(np.uint8) * 255

    # Simple morphological closing via scipy if available, else skip.
    try:
        from scipy.ndimage import binary_closing
        struct = np.ones((15, 15), dtype=bool)
        fg = (binary_closing(fg > 0, structure=struct).astype(np.uint8)) * 255
    except ImportError:
        pass

    # Fallback: if almost nothing detected, treat whole image as foreground
    if fg.sum() / 255 < min_area_frac * fg.size:
        fg[:] = 255

    return fg


def _compute_foreground_coverage(
    mask_array: np.ndarray,
    fg_mask: np.ndarray,
) -> float:
    """Return the fraction of *foreground* pixels that are masked.

    If there is no foreground (all-zero ``fg_mask``), falls back to
    whole-image coverage to avoid division by zero.
    """
    fg_pixels = int(fg_mask.sum()) // 255
    if fg_pixels == 0:
        total = mask_array.size
        return float(np.sum(mask_array > 0) / total) if total else 0.0
    masked_fg = np.sum((mask_array > 0) & (fg_mask > 0))
    return float(masked_fg / fg_pixels)


class MaskingStrategy:
    """
    Encapsulates strategies for generating image masks and creating inpainting datasets.
    """
    def __init__(self, strategy_name: str, mask_config: dict, output_root: Path = None, debug_visualization: bool = False, hero_tracker=None):
        self.mask_config = mask_config
        self.logger = logging.getLogger(__name__)
        self.plotter = ThesisPlotter(output_root) if output_root else None
        self.debug_visualization = debug_visualization
        self.hero_tracker = hero_tracker
        
        # Factory for generators
        if strategy_name == "random_rectangle":
            self.generator = BoxMaskGenerator(
                min_ratio=mask_config.get('box_min_ratio', mask_config.get('min_mask_size_ratio', 0.2)),
                max_ratio=mask_config.get('box_max_ratio', mask_config.get('max_mask_size_ratio', 0.6))
            )
        elif strategy_name == "irregular":
            self.generator = IrregularMaskGenerator(
                min_strokes=mask_config.get('irregular_min_strokes', 3),
                max_strokes=mask_config.get('irregular_max_strokes', 10),
                min_width=mask_config.get('irregular_min_width', 20),
                max_width=mask_config.get('irregular_max_width', 60),
            )
        elif strategy_name == "edge":
            self.generator = EdgeMaskGenerator(
                depth_ratio=mask_config.get('edge_depth_ratio', 0.25),
            )
        else:
            self.logger.error(f"Unknown masking strategy: {strategy_name}")
            raise ValueError(f"Unknown masking strategy: {strategy_name}")

        self.min_coverage = mask_config.get('min_coverage', 0.15)
        self.max_coverage = mask_config.get('max_coverage', 0.35)
        self.foreground_roi = mask_config.get('foreground_roi', True)
        self.max_retries = mask_config.get('max_retries', 20)
        self.max_masks_per_image = mask_config.get('max_masks_per_image', 1)

    # Supported extensions (Stage 03 writes JPG crops, Stage 08 converts to PNG)
    _IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg'}

    def create_inpainting_dataset(self, image_dir: Path, output_dir: Path, global_seed: int = 42):
        """
        Creates a dataset for inpainting by generating masks for a set of images.

        Parameters
        ----------
        global_seed : int
            Combined with each image's stem hash to produce a per-image seed.
            This ensures the same mask is always generated for a given image
            regardless of processing order.
        """
        image_files = sorted(
            [p for p in image_dir.iterdir()
             if p.is_file() and p.suffix.lower() in self._IMAGE_EXTENSIONS]
        )
        
        if not image_files:
            self.logger.warning(f"No PNG images found in {image_dir}. Nothing to process.")
            return

        ground_truth_dir = output_dir / "ground_truth"
        masks_dir = output_dir / "masks"
        captions_dir = output_dir / "captions"
        ground_truth_dir.mkdir(parents=True, exist_ok=True)
        masks_dir.mkdir(parents=True, exist_ok=True)
        captions_dir.mkdir(parents=True, exist_ok=True)

        self.logger.info(f"Generating masks for {len(image_files)} images from {image_dir.name} split...")
        self.logger.info(
            f"  min_coverage={self.min_coverage}, max_coverage={self.max_coverage}, "
            f"foreground_roi={self.foreground_roi}, max_retries={self.max_retries}"
        )

        coverage_stats = []
        accumulated_mask = None
        heatmap_size = (256, 256)
        debug_samples = []

        # Determine mask type name for CSV reporting
        generator_class = type(self.generator).__name__
        _TYPE_NAMES = {
            "BoxMaskGenerator": "rect",
            "IrregularMaskGenerator": "irregular",
            "EdgeMaskGenerator": "edge",
            "_DispatchGenerator": "mixed",
        }
        mask_type_label = _TYPE_NAMES.get(generator_class, generator_class)

        for img_path in tqdm(image_files, desc=f"Generating masks for {image_dir.name}"):
            try:
                with Image.open(img_path).convert("RGB") as img:
                    width, height = img.size
                    img_array = np.array(img)

                # Per-image deterministic seed = global_seed XOR hash(stem)
                img_seed = global_seed ^ (hash(img_path.stem) & 0x7FFFFFFF)

                # Generate multiple masks per image for data augmentation
                num_masks = max(1, self.max_masks_per_image)

                # Pre-compute foreground ROI mask once per image
                if self.foreground_roi:
                    fg_mask = _compute_foreground_mask(img_array)
                else:
                    fg_mask = np.full((height, width), 255, dtype=np.uint8)

                for mask_idx in range(num_masks):
                    mask_seed = img_seed + mask_idx * 7919  # Offset seed per mask variant

                    # Generate mask with min/max coverage enforcement (foreground-aware)
                    mask_array = None
                    best_candidate = None
                    best_deviation = float("inf")
                    target_mid = (self.min_coverage + self.max_coverage) / 2.0

                    for retry in range(self.max_retries):
                        candidate = self.generator.generate(height, width, seed=mask_seed + retry)
                        # Restrict mask to foreground region
                        candidate = candidate & fg_mask
                        fg_cov = _compute_foreground_coverage(candidate, fg_mask)

                        if self.min_coverage <= fg_cov <= self.max_coverage:
                            mask_array = candidate
                            break

                        # Track closest-to-target for fallback
                        deviation = abs(fg_cov - target_mid)
                        if deviation < best_deviation:
                            best_deviation = deviation
                            best_candidate = candidate

                    if mask_array is None:
                        if best_candidate is None:
                            raise RuntimeError(
                                f"Mask for {img_path.name} (variant {mask_idx}) could not be generated "
                                f"after {self.max_retries} retries."
                            )
                        mask_array = best_candidate
                        final_cov = _compute_foreground_coverage(mask_array, fg_mask)
                        self.logger.warning(
                            f"Mask for {img_path.name} (variant {mask_idx}) could not achieve "
                            f"coverage in [{self.min_coverage:.0%}, {self.max_coverage:.0%}] after "
                            f"{self.max_retries} retries; using closest candidate at {final_cov:.1%}."
                        )

                    coverage = _compute_foreground_coverage(mask_array, fg_mask)
                    # Also record whole-image coverage for stats
                    whole_cov = float(np.sum(mask_array > 0) / (width * height))
                    coverage_stats.append({
                        "coverage_ratio": whole_cov,
                        "fg_coverage_ratio": coverage,
                        "mask_type": mask_type_label,
                    })

                    # Determine output filename: original name for first mask, suffixed for extras
                    if num_masks == 1:
                        out_name = img_path.name
                        out_stem = img_path.stem
                    else:
                        out_name = f"{img_path.stem}_m{mask_idx}{img_path.suffix}"
                        out_stem = f"{img_path.stem}_m{mask_idx}"

                    # Copy ground truth (only once per image, for first mask variant)
                    if mask_idx == 0:
                        if num_masks == 1:
                            shutil.copy(img_path, ground_truth_dir / img_path.name)
                        else:
                            # For multi-mask, create a copy per variant so dataset stays aligned
                            shutil.copy(img_path, ground_truth_dir / out_name)
                    else:
                        shutil.copy(img_path, ground_truth_dir / out_name)

                    mask_image = Image.fromarray(mask_array, mode='L')
                    mask_image.save(masks_dir / out_name)

                    # Resize for heatmap
                    mask_resized = np.array(mask_image.resize(heatmap_size)) / 255.0
                    if accumulated_mask is None:
                        accumulated_mask = mask_resized
                    else:
                        accumulated_mask += mask_resized

                    # Collect Debug Samples (only first variant)
                    if mask_idx == 0:
                        masked_visual = img_array.copy()
                        masked_visual[mask_array > 0] = 0

                        if self.debug_visualization and len(debug_samples) < 4:
                            debug_samples.append((img_array, mask_array, masked_visual))

                        # Hero Tracking
                        if self.hero_tracker:
                            self.hero_tracker.log_image(masked_visual, "08_masked_input", img_path.name)
                            self.hero_tracker.log_image(mask_array, "08_mask", img_path.name)

                    # Copy Caption
                    txt_src = image_dir / f"{img_path.stem}.txt"
                    if txt_src.exists():
                        shutil.copy(txt_src, captions_dir / f"{out_stem}.txt")

            except Exception as e:
                self.logger.error(f"Failed to process {img_path}: {e}")

        # Generate Visualizations via Plotter
        if self.plotter:
            self._save_reports(image_dir.name, coverage_stats, accumulated_mask, debug_samples)

        self.logger.info(f"Finished generating masks for the {image_dir.name} split.")

    def _save_reports(self, split_name, coverage_stats, accumulated_mask, debug_samples):
        if coverage_stats:
            df_coverage = pd.DataFrame(coverage_stats)
            df_coverage.to_csv(self.plotter.output_dir / f"mask_coverage_{split_name}.csv", index=False)
            
            coverage_values = df_coverage["coverage_ratio"]
            self.plotter.plot_histogram(
                coverage_values, 
                f"Mask Coverage Distribution ({split_name})", 
                "Masked Percentage", 
                f"mask_coverage_hist_{split_name}",
                color='danger'
            )
        
        if accumulated_mask is not None:
            self.plotter.plot_heatmap(
                accumulated_mask / len(coverage_stats), 
                f"Damage Heatmap ({split_name})",
                f"damage_heatmap_{split_name}"
            )
        
        if self.debug_visualization and debug_samples:
            self._save_debug_grid(debug_samples, self.plotter.output_dir, split_name)

    def _save_debug_grid(self, samples, output_dir, split_name):
        import matplotlib.pyplot as plt
        n_samples = min(len(samples), 4)
        fig, axes = plt.subplots(n_samples, 3, figsize=(12, 4 * n_samples))
        if n_samples == 1: axes = axes.reshape(1, -1)

        for i in range(n_samples):
            orig, mask, masked = samples[i]
            axes[i, 0].imshow(orig); axes[i, 0].set_title("Original"); axes[i, 0].axis("off")
            axes[i, 1].imshow(mask, cmap="gray"); axes[i, 1].set_title("Mask"); axes[i, 1].axis("off")
            axes[i, 2].imshow(masked); axes[i, 2].set_title("Masked Input"); axes[i, 2].axis("off")

        plt.tight_layout()
        plt.savefig(output_dir / f"masking_preview_{split_name}.png", dpi=150)
        plt.close()