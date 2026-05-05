# src/thesis_pipeline/components/splitting.py
import re
import logging
import shutil
import pandas as pd
from pathlib import Path
from collections import defaultdict
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from thesis_pipeline.visualization import ThesisPlotter

# Supported image extensions (Stage 03 writes JPG, Stage 08 writes PNG)
_IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg'}


class DataSplitter:
    """
    Splits data into train / validation / test sets **by source image**,
    preventing data leakage from crops of the same original image.

    Stage 03 generates crops with names like ``vase1_crop0.jpg``,
    ``vase1_crop1.jpg``.  A naive stem-level split could place different crops
    of the same source image into different splits, leaking information.

    This splitter:
      1. Groups files by their **source image** (strip ``_crop\\d+`` suffix).
      2. Splits *source IDs* (not individual files).
      3. Expands back to filenames after splitting.
      4. Runs a post-split leakage check.
    """

    _CROP_SUFFIX_RE = re.compile(r'_crop\d+$')

    def __init__(self, input_dir: Path, output_dir: Path, report_dir: Path,
                 test_size: float, validation_size: float, random_state: int):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.test_size = test_size
        self.validation_size = validation_size
        self.random_state = random_state
        self.logger = logging.getLogger(__name__)
        self.plotter = ThesisPlotter(report_dir)

        # --- Ratio validation ---
        if not (0 < test_size < 1):
            raise ValueError(f"test_size must be in (0, 1), got {test_size}")
        if not (0 < validation_size < 1):
            raise ValueError(f"validation_size must be in (0, 1), got {validation_size}")
        if test_size + validation_size >= 1.0:
            raise ValueError(
                f"test_size ({test_size}) + validation_size ({validation_size}) "
                f"= {test_size + validation_size} >= 1.0 — nothing left for training."
            )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _source_id(self, stem: str) -> str:
        """Strip ``_crop\\d+`` suffix to get the source image identifier."""
        return self._CROP_SUFFIX_RE.sub('', stem)

    def _find_all_files(self) -> dict:
        """Return ``{source_id: [stem, stem, …]}`` for all image files."""
        groups: dict[str, list[str]] = defaultdict(list)
        seen_stems: set[str] = set()
        for p in self.input_dir.iterdir():
            if p.suffix.lower() in _IMAGE_EXTENSIONS and p.is_file():
                stem = p.stem
                if stem not in seen_stems:
                    seen_stems.add(stem)
                    groups[self._source_id(stem)].append(stem)
        self.logger.info(
            f"Found {len(seen_stems)} files from {len(groups)} unique source images "
            f"in {self.input_dir}."
        )
        return dict(groups)

    def _copy_files(self, stems: list, destination_dir: Path):
        """Copy images (any supported extension) and associated metadata."""
        destination_dir.mkdir(parents=True, exist_ok=True)
        for stem in tqdm(stems, desc=f"Copying to {destination_dir.name}"):
            copied = False
            for ext in _IMAGE_EXTENSIONS:
                img_src = self.input_dir / f"{stem}{ext}"
                if img_src.exists():
                    try:
                        shutil.copy(img_src, destination_dir / img_src.name)
                        copied = True
                    except Exception as e:
                        self.logger.error(f"Could not copy image {img_src}: {e}")
                    break  # only one extension per stem
            if not copied:
                self.logger.warning(f"No image file found for stem '{stem}' — skipped.")

            # Copy associated metadata / caption
            txt_src = self.input_dir / f"{stem}.txt"
            if txt_src.exists():
                try:
                    shutil.copy(txt_src, destination_dir / txt_src.name)
                except Exception as e:
                    self.logger.error(f"Could not copy metadata {txt_src}: {e}")

    # ------------------------------------------------------------------
    # Leakage check
    # ------------------------------------------------------------------

    def _check_leakage(self, train_stems, val_stems, test_stems):
        """Verify zero source-image overlap across splits. Raise on leak."""
        train_sources = {self._source_id(s) for s in train_stems}
        val_sources = {self._source_id(s) for s in val_stems}
        test_sources = {self._source_id(s) for s in test_stems}

        tv = train_sources & val_sources
        tt = train_sources & test_sources
        vt = val_sources & test_sources

        if tv or tt or vt:
            msg = (
                "DATA LEAKAGE DETECTED after splitting!\n"
                f"  train∩val  : {len(tv)} sources — {list(tv)[:5]}\n"
                f"  train∩test : {len(tt)} sources — {list(tt)[:5]}\n"
                f"  val∩test   : {len(vt)} sources — {list(vt)[:5]}"
            )
            self.logger.critical(msg)
            raise RuntimeError(msg)

        self.logger.info(
            "Leakage check PASSED — zero source-image overlap between "
            f"train ({len(train_sources)}), val ({len(val_sources)}), "
            f"test ({len(test_sources)}) source IDs."
        )

    # ------------------------------------------------------------------
    # Main entry
    # ------------------------------------------------------------------

    def split_data(self):
        """Split image files into train / val / test and copy them."""
        source_groups = self._find_all_files()
        source_ids = sorted(source_groups.keys())

        if not source_ids:
            self.logger.warning("No files found in the input directory. Nothing to split.")
            return

        # --- Split on SOURCE IDs (not individual files) ---
        self.logger.info(f"Splitting {len(source_ids)} source images "
                         f"(test={self.test_size}, val={self.validation_size}).")

        train_val_ids, test_ids = train_test_split(
            source_ids,
            test_size=self.test_size,
            random_state=self.random_state,
        )

        val_size_adjusted = self.validation_size / (1.0 - self.test_size)
        train_ids, val_ids = train_test_split(
            train_val_ids,
            test_size=val_size_adjusted,
            random_state=self.random_state,
        )

        # --- Expand source IDs back to individual file stems ---
        train_stems = [s for sid in train_ids for s in source_groups[sid]]
        val_stems = [s for sid in val_ids for s in source_groups[sid]]
        test_stems = [s for sid in test_ids for s in source_groups[sid]]

        self.logger.info("Split complete (source → file counts):")
        self.logger.info(f"  Train : {len(train_ids)} sources → {len(train_stems)} files")
        self.logger.info(f"  Val   : {len(val_ids)} sources → {len(val_stems)} files")
        self.logger.info(f"  Test  : {len(test_ids)} sources → {len(test_stems)} files")

        # --- Leakage safety check ---
        self._check_leakage(train_stems, val_stems, test_stems)

        # --- Visualize ---
        data = {
            'Split': ['Train', 'Validation', 'Test'],
            'Count': [len(train_stems), len(val_stems), len(test_stems)]
        }
        df = pd.DataFrame(data)
        df.to_csv(self.plotter.output_dir / "split_stats.csv", index=False)
        self.plotter.plot_bar(
            df['Split'], df['Count'],
            "Data Split Distribution", "Split", "Sample Count",
            "split_distribution",
        )

        # --- Copy files ---
        train_dir = self.output_dir / 'train'
        val_dir = self.output_dir / 'validation'
        test_dir = self.output_dir / 'test'

        self.logger.info("Copying files to split directories…")
        self._copy_files(train_stems, train_dir)
        self._copy_files(val_stems, val_dir)
        self._copy_files(test_stems, test_dir)

        self.logger.info("Data splitting stage finished.")