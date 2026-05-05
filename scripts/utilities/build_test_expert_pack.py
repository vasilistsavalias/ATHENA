"""Build a two-pair Expert_Pack_v2 from archived Stage 13 outputs.

Block B layout (per Codex recommendation):
  - 10 items: FT-SD vs Telea    (classical baseline)
  - 10 items: FT-SD vs LaMa     (strong deep baseline)

Block A layout:
  - 20 items: mix of real (ground truth) and generated (FT-SD/Telea/LaMa)

Pair type is encoded in sample_id prefix:
  - "PAIR_FTSD_TELEA__<original_id>"
  - "PAIR_FTSD_LAMA__<original_id>"
This lets downstream analysis split by prefix with zero schema changes.

Usage:
    python scripts/utilities/build_test_expert_pack.py

Creates:
    archieve/old_outputs_v5/13_model_evaluation/Expert_Pack_v2/
    archieve/old_outputs_v5/13_model_evaluation/stage13_ground_truth/

Import into website DB:
    cd website/services/api
    python -m app.tools.import_pack \\
        --pack-dir "../../archieve/old_outputs_v5/13_model_evaluation/Expert_Pack_v2" \\
        --stage13-samples "../../archieve/old_outputs_v5/13_model_evaluation/stage13_ground_truth" \\
        --campaign-name "Test Wave (Archive)"
"""

from __future__ import annotations

import json
import random
import shutil
from datetime import datetime, timezone
from hashlib import md5
from pathlib import Path

import pandas as pd

# ----- Config -----
ARCHIVE_ROOT = Path("archieve/old_outputs_v5/13_model_evaluation")
SAMPLES_DIR = ARCHIVE_ROOT / "samples"
MATRIX_CSV = ARCHIVE_ROOT / "benchmarking_matrix" / "matrix_results.csv"
OUTPUT_PACK = ARCHIVE_ROOT / "Expert_Pack_v2"
OUTPUT_GT = ARCHIVE_ROOT / "stage13_ground_truth"

# Two comparison pairs
PAIRS = [
    {
        "tag": "PAIR_FTSD_TELEA",
        "method_a": "Telea",
        "method_b": "FT-SD",
        "file_a": "Telea_Unconditional.png",
        "file_b": "FT_SD_Unconditional.png",
        "count": 10,   # Block B items for this pair
    },
    {
        "tag": "PAIR_FTSD_LAMA",
        "method_a": "LaMa",
        "method_b": "FT-SD",
        "file_a": "LaMa_Unconditional.png",
        "file_b": "FT_SD_Unconditional.png",
        "count": 10,   # Block B items for this pair
    },
]
BLOCK_A_EXTRA = 20  # Extra samples for Block A disjoint pool (real + generated)
CONDITION = "Unconditional"
SEED = 42


def _load_metadata() -> dict[str, dict]:
    """Load mask_type/coverage metadata from matrix CSV, keyed by folder name."""
    df = pd.read_csv(MATRIX_CSV)
    df = df[df["condition"] == "Unconditional"].copy()
    meta_df = df.groupby("sample_id", as_index=False)[["mask_type", "mask_coverage"]].first()
    meta_df["coverage_bin"] = pd.cut(
        meta_df["mask_coverage"],
        bins=[0.0, 0.10, 0.25, 0.50, 1.0],
        labels=["<10%", "10-25%", "25-50%", ">50%"],
        include_lowest=True,
    ).astype(str)

    lookup: dict[str, dict] = {}
    for _, row in meta_df.iterrows():
        csv_id = str(row["sample_id"])
        folder_id = csv_id.replace(".png", "") if csv_id.endswith(".png") else csv_id
        lookup[folder_id] = {
            "mask_type": row["mask_type"],
            "mask_coverage": float(row["mask_coverage"]),
            "coverage_bin": row["coverage_bin"],
        }
    return lookup


def _stratified_select(
    candidates: list[str],
    meta_lookup: dict[str, dict],
    count: int,
    rng: random.Random,
) -> list[str]:
    """Stratified random selection: 1 per (mask_type, coverage_bin), then fill."""
    grouped: dict[tuple[str, str], list[str]] = {}
    for fid in candidates:
        meta = meta_lookup.get(fid)
        if meta:
            key = (meta["mask_type"], meta["coverage_bin"])
            grouped.setdefault(key, []).append(fid)

    selected: list[str] = []
    remaining = count

    for key in sorted(grouped.keys()):
        if remaining <= 0:
            break
        pool = grouped[key][:]
        rng.shuffle(pool)
        selected.append(pool[0])
        remaining -= 1

    if remaining > 0:
        pool = sorted(set(candidates) - set(selected))
        rng.shuffle(pool)
        selected.extend(pool[:remaining])

    return selected


def main():
    rng = random.Random(SEED)
    meta_lookup = _load_metadata()

    # --- Find samples valid for ALL pairs ---
    all_required_files = set()
    for pair in PAIRS:
        all_required_files.add(pair["file_a"])
        all_required_files.add(pair["file_b"])
    all_required_files.add("masked_input.png")

    all_folders = sorted([d.name for d in SAMPLES_DIR.iterdir() if d.is_dir()])
    valid_folders = []
    for folder in all_folders:
        src = SAMPLES_DIR / folder
        if all(( src / f).exists() for f in all_required_files):
            valid_folders.append(folder)

    print(f"Valid samples (all methods present): {len(valid_folders)}")

    # --- Allocate samples to pairs (non-overlapping within Block B) ---
    total_block_b = sum(p["count"] for p in PAIRS)
    total_needed = total_block_b + BLOCK_A_EXTRA
    print(f"Need: {total_block_b} Block B + {BLOCK_A_EXTRA} Block A extra = {total_needed} total")

    # Select the full pool with stratification
    full_pool = _stratified_select(valid_folders, meta_lookup, total_needed, rng)
    rng.shuffle(full_pool)

    # Partition: first chunk for Block A disjoint, rest split across pairs
    block_a_pool = full_pool[:BLOCK_A_EXTRA]
    block_b_pool = full_pool[BLOCK_A_EXTRA:]

    pair_allocations: dict[str, list[str]] = {}
    offset = 0
    for pair in PAIRS:
        pair_allocations[pair["tag"]] = block_b_pool[offset : offset + pair["count"]]
        offset += pair["count"]

    # --- Clean output dirs ---
    if OUTPUT_PACK.exists():
        shutil.rmtree(OUTPUT_PACK)
    if OUTPUT_GT.exists():
        shutil.rmtree(OUTPUT_GT)
    OUTPUT_PACK.mkdir(parents=True)
    OUTPUT_GT.mkdir(parents=True)
    images_root = OUTPUT_PACK / "images"
    images_root.mkdir()

    # --- Build Block B items (with pair-tagged sample_ids) ---
    public_items = []
    private_items = []

    for pair in PAIRS:
        tag = pair["tag"]
        allocated = pair_allocations[tag]
        print(f"  {tag}: {len(allocated)} items")

        for sid in allocated:
            tagged_id = f"{tag}__{sid}"
            src_dir = SAMPLES_DIR / sid
            dest_dir = images_root / tagged_id
            dest_dir.mkdir(parents=True)

            shutil.copy2(src_dir / "masked_input.png", dest_dir / "input.png")

            # Blinded A/B assignment (deterministic per tagged sample)
            sid_hash = int(md5(tagged_id.encode("utf-8", errors="ignore")).hexdigest()[:8], 16)
            local_rng = random.Random(SEED ^ sid_hash)

            order = [
                ("A", pair["method_a"], src_dir / pair["file_a"]),
                ("B", pair["method_b"], src_dir / pair["file_b"]),
            ]
            local_rng.shuffle(order)

            mapping = {}
            for label, method_name, path in order:
                shutil.copy2(path, dest_dir / f"{label}.png")
                mapping[label] = method_name

            meta = meta_lookup.get(sid, {})
            public_items.append({
                "sample_id": tagged_id,
                "input": f"images/{tagged_id}/input.png",
                "A": f"images/{tagged_id}/A.png",
                "B": f"images/{tagged_id}/B.png",
                "pair_tag": tag,
                **meta,
            })
            private_items.append({
                "sample_id": tagged_id,
                "pair_tag": tag,
                "original_sample_id": sid,
                "mapping": mapping,
            })

    # --- Block A disjoint samples (also add as pack items for generated images) ---
    for sid in block_a_pool:
        tagged_id = f"BLOCKA__{sid}"
        src_dir = SAMPLES_DIR / sid
        dest_dir = images_root / tagged_id
        dest_dir.mkdir(parents=True)

        # Use FT-SD and Telea as the two generated variants for Block A
        shutil.copy2(src_dir / "masked_input.png", dest_dir / "input.png")
        shutil.copy2(src_dir / "FT_SD_Unconditional.png", dest_dir / "A.png")
        shutil.copy2(src_dir / "Telea_Unconditional.png", dest_dir / "B.png")

        meta = meta_lookup.get(sid, {})
        public_items.append({
            "sample_id": tagged_id,
            "input": f"images/{tagged_id}/input.png",
            "A": f"images/{tagged_id}/A.png",
            "B": f"images/{tagged_id}/B.png",
            "pair_tag": "BLOCKA_ONLY",
            **meta,
        })
        private_items.append({
            "sample_id": tagged_id,
            "pair_tag": "BLOCKA_ONLY",
            "original_sample_id": sid,
            "mapping": {"A": "FT-SD", "B": "Telea"},
        })

        # Copy ground truth for "real" Block A images
        original = src_dir / "original.png"
        if original.exists():
            gt_dest = OUTPUT_GT / tagged_id
            gt_dest.mkdir(parents=True)
            shutil.copy2(original, gt_dest / "ground_truth.png")

    # --- Write manifests ---
    created_at = datetime.now(timezone.utc).isoformat()
    rating_schema = {
        "task": "Pick the better restoration for the missing region (historical plausibility + visual coherence).",
        "fields": [
            {"name": "choice", "type": "enum", "values": ["A", "B", "Tie", "Unsure"]},
            {"name": "confidence", "type": "int", "min": 1, "max": 5},
            {"name": "comment", "type": "string", "optional": True},
        ],
    }

    manifest_public = {
        "pack_id": "Expert_Pack_v2",
        "created_at": created_at,
        "pairs": [
            {
                "tag": p["tag"],
                "method_pair": [p["method_a"], p["method_b"]],
                "condition": CONDITION,
                "count": p["count"],
            }
            for p in PAIRS
        ],
        "rating_schema": rating_schema,
        "items": public_items,
    }
    manifest_private = {
        "pack_id": "Expert_Pack_v2",
        "created_at": created_at,
        "items": private_items,
    }

    (OUTPUT_PACK / "manifest_public.json").write_text(
        json.dumps(manifest_public, indent=2), encoding="utf-8"
    )
    (OUTPUT_PACK / "manifest_private.json").write_text(
        json.dumps(manifest_private, indent=2), encoding="utf-8"
    )

    # --- Summary ---
    print(f"\nPack created at: {OUTPUT_PACK}")
    print(f"Ground truth dir: {OUTPUT_GT}")
    print(f"Total manifest items: {len(public_items)}")
    print(f"  Block B (FT-SD vs Telea): {PAIRS[0]['count']}")
    print(f"  Block B (FT-SD vs LaMa):  {PAIRS[1]['count']}")
    print(f"  Block A pool (disjoint):  {len(block_a_pool)}")
    print(f"Ground truth images: {sum(1 for d in OUTPUT_GT.iterdir() if d.is_dir())}")

    # Coverage breakdown
    block_b_items = [i for i in public_items if i.get("pair_tag") != "BLOCKA_ONLY"]
    bins = {}
    for item in block_b_items:
        b = item.get("coverage_bin", "unknown")
        bins[b] = bins.get(b, 0) + 1
    print(f"Block B coverage distribution: {bins}")

    types = {}
    for item in block_b_items:
        t = item.get("mask_type", "unknown")
        types[t] = types.get(t, 0) + 1
    print(f"Block B mask type distribution: {types}")


if __name__ == "__main__":
    main()
