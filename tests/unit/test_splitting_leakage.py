"""Tests that DataSplitter prevents source-image leakage across splits.

Creates mock crops like ``vase1_crop0.png``, ``vase1_crop1.png``, etc. and
verifies that all crops from one source image stay in the same split.
"""
import unittest
import re
import tempfile
import shutil
from pathlib import Path
from PIL import Image

from thesis_pipeline.components.splitting import DataSplitter


class TestSplittingLeakage(unittest.TestCase):
    """Verify that crop-level data leakage is impossible after the fix."""

    CROP_RE = re.compile(r'_crop\d+$')

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.input_dir = self.tmp / "input"
        self.output_dir = self.tmp / "splits"
        self.report_dir = self.tmp / "report"
        self.input_dir.mkdir()
        self.output_dir.mkdir()
        self.report_dir.mkdir()

        # Create 10 source images × 3 crops each = 30 files
        for i in range(10):
            for c in range(3):
                img = Image.new("RGB", (64, 64), color=(i * 25, c * 80, 128))
                img.save(self.input_dir / f"source{i:02d}_crop{c}.png")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _source_id(self, stem: str) -> str:
        return self.CROP_RE.sub('', stem)

    def test_no_source_overlap_between_splits(self):
        splitter = DataSplitter(
            input_dir=self.input_dir,
            output_dir=self.output_dir,
            report_dir=self.report_dir,
            test_size=0.2,
            validation_size=0.1,
            random_state=42,
        )
        splitter.split_data()

        # Collect source IDs per split
        split_sources = {}
        for split in ["train", "validation", "test"]:
            d = self.output_dir / split
            if d.exists():
                stems = {p.stem for p in d.glob("*.png")}
                sources = {self._source_id(s) for s in stems}
                split_sources[split] = sources

        # Verify pairwise disjoint
        splits = list(split_sources.keys())
        for i, a in enumerate(splits):
            for b in splits[i + 1:]:
                overlap = split_sources[a] & split_sources[b]
                self.assertEqual(
                    len(overlap), 0,
                    f"Source images leaked between {a} and {b}: {overlap}"
                )

    def test_all_crops_of_source_in_same_split(self):
        splitter = DataSplitter(
            input_dir=self.input_dir,
            output_dir=self.output_dir,
            report_dir=self.report_dir,
            test_size=0.2,
            validation_size=0.1,
            random_state=42,
        )
        splitter.split_data()

        # For each source, all its crops must be in exactly one split
        for i in range(10):
            source = f"source{i:02d}"
            found_in = set()
            for split in ["train", "validation", "test"]:
                d = self.output_dir / split
                for c in range(3):
                    if (d / f"{source}_crop{c}.png").exists():
                        found_in.add(split)
            self.assertEqual(
                len(found_in), 1,
                f"Source '{source}' found in multiple splits: {found_in}"
            )

    def test_invalid_ratios_raise(self):
        with self.assertRaises(ValueError):
            DataSplitter(
                input_dir=self.input_dir,
                output_dir=self.output_dir,
                report_dir=self.report_dir,
                test_size=0.6,
                validation_size=0.5,  # sum = 1.1
                random_state=42,
            )


if __name__ == "__main__":
    unittest.main()
