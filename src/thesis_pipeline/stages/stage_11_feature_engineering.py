# src/thesis_pipeline/pipeline/stage_11_feature_engineering.py
import json
import logging
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from thesis_pipeline.config_manager import ConfigManager
from thesis_pipeline.components.masking import MaskingStrategy
from thesis_pipeline.utils.hero_tracker import HeroTracker
from thesis_pipeline.utils.seed_manager import SeedManager
from thesis_pipeline.utils.stage_artifacts import resolve_stage_artifact_dir

class FeatureEngineeringStage:
    def __init__(self, config_manager: ConfigManager):
        self.config_manager = config_manager
        self.config = config_manager.get_feature_engineering_config()
        self.hero_config = config_manager.config.get("hero_tracking")
        self.paths = config_manager.get_paths()
        self.logger = logging.getLogger(__name__)
        self.output_dir = Path(self.paths.data.inpainting)
        self.report_dir = resolve_stage_artifact_dir(self.config_manager, "S11")
        # Coerce to a stable int seed even when config_manager is a mock (tests).
        try:
            self.global_seed = int(SeedManager.get_seed_from_config(config_manager))
        except Exception:
            self.global_seed = 42

    def run(self):
        self.logger.info("="*20 + " STAGE S11: Feature Engineering + Mask Realism " + "="*20)
        try:
            input_dir = Path(self.paths.data.splits)
            self.report_dir.mkdir(parents=True, exist_ok=True)
            self.output_dir.mkdir(parents=True, exist_ok=True)
            
            hero_tracker = None
            if self.hero_config and self.hero_config.get('enabled'):
                hero_tracker = HeroTracker(Path(self.hero_config.output_dir), self.hero_config.hero_filenames)

            # --- Resolve mask types ---
            mask_types = self.config.get("mask_types")
            if not mask_types:
                mask_types = [self.config.get("mask_strategy", "random_rectangle")]

            # Normalise legacy config names → generator names
            _ALIAS = {"rect": "random_rectangle", "irregular": "irregular", "edge": "edge",
                       "random_rectangle": "random_rectangle"}
            mask_types = [_ALIAS.get(t, t) for t in mask_types]

            mask_config = self.config.get("mask_config", self.config.to_dict())

            # Build one MaskingStrategy per type
            strategies = {}
            for mt in mask_types:
                strategies[mt] = MaskingStrategy(
                    strategy_name=mt,
                    mask_config=mask_config,
                    output_root=self.report_dir,
                    debug_visualization=self.config.get("debug_visualization", False),
                    hero_tracker=hero_tracker,
                )

            self.logger.info(f"Mask strategies loaded: {list(strategies.keys())}")

            # --- If only one strategy, delegate directly (fast path) ---
            if len(strategies) == 1:
                sole_strategy = next(iter(strategies.values()))
                stats = {}
                for split in ["train", "validation", "test"]:
                    split_input = input_dir / split
                    split_output = self.output_dir / split
                    if not split_input.exists():
                        stats[split] = 0
                        continue
                    split_output.mkdir(parents=True, exist_ok=True)
                    sole_strategy.create_inpainting_dataset(
                        split_input, split_output, global_seed=self.global_seed,
                    )
                    mask_count = len(list((split_output / "masks").glob("*.png")))
                    stats[split] = mask_count
            else:
                # --- Multiple strategies: assign each image a type deterministically ---
                stats = {}
                type_counts = {mt: 0 for mt in mask_types}
                for split in ["train", "validation", "test"]:
                    split_input = input_dir / split
                    split_output = self.output_dir / split
                    if not split_input.exists():
                        stats[split] = 0
                        continue
                    split_output.mkdir(parents=True, exist_ok=True)

                    # We need per-image control, so we iterate images ourselves.
                    # Collect image files (matching what MaskingStrategy would use).
                    from thesis_pipeline.components.masking import MaskingStrategy as _MS
                    image_files = sorted(
                        [p for p in split_input.iterdir()
                         if p.is_file() and p.suffix.lower() in _MS._IMAGE_EXTENSIONS]
                    )

                    # Pre-assign strategy per image (deterministic via seed + stem hash)
                    for img_path in image_files:
                        img_seed = self.global_seed ^ (hash(img_path.stem) & 0x7FFFFFFF)
                        chosen_type = mask_types[img_seed % len(mask_types)]
                        type_counts[chosen_type] = type_counts.get(chosen_type, 0) + 1

                    # Delegate full dataset creation to first strategy;
                    # Override: use a MixedMaskingStrategy wrapper below.
                    # Actually simplest correct approach: call each strategy on its
                    # assigned subset.  But MaskingStrategy.create_inpainting_dataset
                    # also copies ground-truth and captions, which we only want once.
                    # So: just use one strategy and override its generator per-image.
                    # Cleanest: use first strategy's create_inpainting_dataset and
                    # monkey-patch generator.generate → dispatch by type.  Too hacky.
                    #
                    # Better: let each strategy process the full directory (its generator
                    # is per-image seeded).  We pick ONE strategy per split by routing
                    # individual files.  For simplicity, we just use the first strategy
                    # but randomly rotate the generator inside.

                    # ---- Simple correct approach: for each split we call
                    # create_inpainting_dataset once (it handles copies, stats, etc.)
                    # with a _DispatchGenerator that wraps all generators.

                    # Capture the *original* underlying generators before we swap the driver's
                    # generator to a dispatcher. Otherwise, dispatching to the driver's own
                    # type can recurse (driver.generator -> dispatch -> driver.generator ...).
                    base_generators = {mt: strategies[mt].generator for mt in mask_types}

                    class _DispatchGenerator:
                        """Dispatches to the right base generator based on image seed."""

                        def __init__(self, generators_by_type, types):
                            self.generators_by_type = generators_by_type
                            self.types = types

                        def generate(self, height, width, *, seed=None):
                            idx = (seed or 0) % len(self.types)
                            chosen_type = self.types[idx]
                            return self.generators_by_type[chosen_type].generate(height, width, seed=seed)

                    dispatch = _DispatchGenerator(base_generators, mask_types)

                    # Use first strategy as the "driver" but swap its generator
                    driver = next(iter(strategies.values()))
                    original_gen = driver.generator
                    driver.generator = dispatch

                    driver.create_inpainting_dataset(
                        split_input, split_output, global_seed=self.global_seed,
                    )

                    # Restore
                    driver.generator = original_gen

                    mask_count = len(list((split_output / "masks").glob("*.png")))
                    stats[split] = mask_count

                stats["mask_type_distribution"] = type_counts
                self.logger.info(f"Mask type distribution: {type_counts}")

            with open(self.report_dir / "masking_stats.json", "w") as f:
                json.dump(stats, f, indent=4)

            self._run_mask_realism_guardrails()

            # --- Generate sample triplets for human verification ---
            self._generate_sample_triplets(input_dir)

            self.logger.info("="*20 + " STAGE S11 COMPLETED " + "="*20 + "\n")
        except Exception as e:
            self.logger.exception(f"Error in Masking stage: {e}")
            raise

    def _run_mask_realism_guardrails(self) -> None:
        """Internal replacement for the former S10b realism stage."""
        cov_path = self.report_dir / "mask_coverage_train.csv"
        if not cov_path.exists():
            self.logger.warning(f"Mask realism guardrails skipped; missing {cov_path}")
            return
        try:
            df = pd.read_csv(cov_path)
            coverage_col = "fg_coverage_ratio" if "fg_coverage_ratio" in df.columns else "coverage_ratio"
            if coverage_col not in df.columns or df[coverage_col].dropna().empty:
                result = {"status": "missing_data", "passed": False, "reason": "coverage data missing"}
            else:
                cov = df[coverage_col].dropna()
                median = float(cov.median())
                p90 = float(cov.quantile(0.90))
                passed = 0.22 <= median <= 0.35 and p90 <= 0.55
                result = {
                    "status": "ok" if passed else "violations",
                    "passed": passed,
                    "median_coverage": median,
                    "p90_coverage": p90,
                    "coverage_basis": coverage_col,
                }
            (self.report_dir / "mask_realism_guardrails.json").write_text(
                json.dumps(result, indent=2), encoding="utf-8"
            )
        except Exception as exc:
            self.logger.warning(f"Mask realism guardrail evaluation failed: {exc}")

    def _generate_sample_triplets(self, input_dir):
        """Save 10 sample triplets (original, mask, masked overlay) per split for verification."""
        from PIL import Image
        import numpy as np

        samples_dir = self.report_dir / "sample_triplets"
        n_samples = 10

        for split in ["train", "validation", "test"]:
            split_output = self.output_dir / split
            gt_dir = split_output / "ground_truth"
            masks_dir = split_output / "masks"

            if not gt_dir.exists() or not masks_dir.exists():
                continue

            split_samples_dir = samples_dir / split
            split_samples_dir.mkdir(parents=True, exist_ok=True)

            gt_files = sorted(gt_dir.glob("*.png")) + sorted(gt_dir.glob("*.jpg"))
            selected = gt_files[:n_samples]

            for gt_path in selected:
                mask_path = masks_dir / gt_path.name
                if not mask_path.exists():
                    # Try alternate extension
                    for ext in [".png", ".jpg", ".jpeg"]:
                        alt = masks_dir / f"{gt_path.stem}{ext}"
                        if alt.exists():
                            mask_path = alt
                            break

                if not mask_path.exists():
                    continue

                try:
                    img = np.array(Image.open(gt_path).convert("RGB"))
                    mask = np.array(Image.open(mask_path).convert("L"))

                    # Create overlay: red tint on masked region
                    overlay = img.copy()
                    overlay[mask > 0] = [255, 0, 0]
                    blended = (img * 0.5 + overlay * 0.5).astype(np.uint8)

                    Image.fromarray(img).save(split_samples_dir / f"{gt_path.stem}_original.png")
                    Image.fromarray(mask).save(split_samples_dir / f"{gt_path.stem}_mask.png")
                    Image.fromarray(blended).save(split_samples_dir / f"{gt_path.stem}_overlay.png")
                except Exception as e:
                    self.logger.warning(f"Failed to create triplet for {gt_path.name}: {e}")

class MaskRealismStage(FeatureEngineeringStage):
    """Compatibility alias; mask realism is now part of FeatureEngineeringStage."""

    _ANALYSIS_TARGET_SIZE = (256, 256)
    _GUARDRAILS = {
        "median_min": 0.22,
        "median_max": 0.35,
        "p90_max": 0.55,
        "min_type_share": 0.20,
    }

    @classmethod
    def evaluate_guardrails(cls, df_coverage):
        coverage_column = None
        if "fg_coverage_ratio" in df_coverage.columns and df_coverage["fg_coverage_ratio"].notna().any():
            coverage_column = "fg_coverage_ratio"
        elif "coverage_ratio" in df_coverage.columns and df_coverage["coverage_ratio"].notna().any():
            coverage_column = "coverage_ratio"

        if df_coverage.empty or coverage_column is None:
            return {
                "status": "missing_data",
                "passed": False,
                "reasons": ["coverage data missing"],
            }

        cov = df_coverage[coverage_column].dropna()
        if cov.empty:
            return {
                "status": "missing_data",
                "passed": False,
                "reasons": [f"{coverage_column} is empty"],
            }

        median = float(cov.median())
        p90 = float(cov.quantile(0.90))

        type_share = {}
        if "mask_type" in df_coverage.columns:
            counts = df_coverage["mask_type"].astype(str).value_counts(normalize=True)
            type_share = {k: float(v) for k, v in counts.to_dict().items()}

        reasons = []
        if median < cls._GUARDRAILS["median_min"] or median > cls._GUARDRAILS["median_max"]:
            reasons.append(
                f"median coverage {median:.4f} outside [{cls._GUARDRAILS['median_min']:.2f}, {cls._GUARDRAILS['median_max']:.2f}]"
            )
        if p90 > cls._GUARDRAILS["p90_max"]:
            reasons.append(f"p90 coverage {p90:.4f} exceeds {cls._GUARDRAILS['p90_max']:.2f}")

        if type_share:
            for mtype, share in type_share.items():
                if share < cls._GUARDRAILS["min_type_share"]:
                    reasons.append(
                        f"mask type '{mtype}' share {share:.4f} below {cls._GUARDRAILS['min_type_share']:.2f}"
                    )

        return {
            "status": "ok" if not reasons else "violations",
            "passed": len(reasons) == 0,
            "coverage_basis": coverage_column,
            "median_coverage": median,
            "p90_coverage": p90,
            "type_share": type_share,
            "reasons": reasons,
            "thresholds": cls._GUARDRAILS,
        }

    def _compute_avg_magnitude_spectrum(self, paths):
        from PIL import Image

        mags = []
        for p in paths:
            mask = np.array(Image.open(p).convert("L"))
            if mask.size == 0:
                continue

            mask_resized = cv2.resize(mask, self._ANALYSIS_TARGET_SIZE, interpolation=cv2.INTER_NEAREST)
            edges = cv2.Canny(mask_resized, 100, 200)

            f = np.fft.fft2(edges)
            fshift = np.fft.fftshift(f)
            magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)
            mags.append(magnitude_spectrum)

        if not mags:
            return None

        return np.mean(np.stack(mags, axis=0), axis=0)


if __name__ == '__main__':
    cm = ConfigManager()
    stage = FeatureEngineeringStage(cm)
    stage.run()
