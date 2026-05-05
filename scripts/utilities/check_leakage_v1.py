"""Check for data leakage across train / validation / test splits.

Performs two levels of checking:
  1. **Filename identity** — exact same filename in multiple splits.
  2. **Source-image overlap** — crops from the same original image
     (``vase1_crop0``, ``vase1_crop1``) appearing in different splits.
"""
import os
import re
import sys


def _source_id(filename: str) -> str:
    """Strip ``_crop\\d+`` suffix and file extension to get source image ID."""
    stem = os.path.splitext(filename)[0]
    return re.sub(r'_crop\d+$', '', stem)


def check_leakage(data_root: str = "data/intermediate/08_inpainting"):
    splits = ["train", "validation", "test"]
    split_files: dict[str, set[str]] = {}
    split_sources: dict[str, set[str]] = {}

    for split in splits:
        gt_dir = os.path.join(data_root, split, "ground_truth")
        if not os.path.exists(gt_dir):
            print(f"  [{split}] directory not found at {gt_dir} — skipping.")
            continue
        files = set(os.listdir(gt_dir))
        split_files[split] = files
        split_sources[split] = {_source_id(f) for f in files}
        print(f"  [{split}] {len(files)} files, {len(split_sources[split])} unique sources.")

    found = splits_present = list(split_files.keys())
    if len(found) < 2:
        print("Need at least 2 splits to check leakage.")
        return

    any_leak = False

    # Pairwise checks
    for i, a in enumerate(found):
        for b in found[i + 1:]:
            # Level 1: filename identity
            overlap_files = split_files[a] & split_files[b]
            if overlap_files:
                print(f"\n  FILENAME LEAK ({a} ∩ {b}): {len(overlap_files)} files")
                print(f"    Examples: {list(overlap_files)[:5]}")
                any_leak = True

            # Level 2: source-image overlap (the critical one)
            overlap_sources = split_sources[a] & split_sources[b]
            if overlap_sources:
                print(f"\n  SOURCE-IMAGE LEAK ({a} ∩ {b}): {len(overlap_sources)} sources")
                print(f"    Examples: {list(overlap_sources)[:5]}")
                any_leak = True

    if not any_leak:
        print("\n  SUCCESS: No data leakage detected across all splits.")
    else:
        print("\n  CRITICAL: Data leakage detected. Fix the splitting logic.")


if __name__ == "__main__":
    root = sys.argv[1] if len(sys.argv) > 1 else "data/intermediate/08_inpainting"
    print(f"Checking leakage in: {root}")
    check_leakage(root)
