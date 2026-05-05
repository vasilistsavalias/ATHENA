"""Build a deterministic website-ready evaluator pack from current Stage 13 outputs.

This utility creates a curated Expert_Pack_v2 that is directly compatible with:
    website/services/api/app/tools/import_pack.py

Design goals:
- Deterministic sample selection (seeded).
- Stratified coverage by mask type + mask coverage bins.
- Disjoint pools by default:
    - First N items are reserved for Block A generated pool.
    - Remaining items are used for Block B paired comparisons.
- Ground-truth copies for Block A real-image pool.

Default selection strategy:
- Block A source samples: 13
- Block B pair 1: FT-SD vs Telea (24)
- Block B pair 2: FT-SD vs LaMa (24)
Total manifest items: 61 (ordered: 13 Block A items first)
"""

from __future__ import annotations

import argparse
import json
import random
import re
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import md5
from pathlib import Path

import pandas as pd


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", str(text)).strip("_")


def _rng(seed: int, salt: str) -> random.Random:
    token = f"{seed}:{salt}".encode("utf-8", errors="ignore")
    mixed = int(md5(token).hexdigest()[:8], 16) ^ int(seed)
    return random.Random(mixed)


def _coverage_bin(value: float) -> str:
    if value <= 0.10:
        return "<10%"
    if value <= 0.25:
        return "10-25%"
    if value <= 0.50:
        return "25-50%"
    return ">50%"


@dataclass(frozen=True)
class PairSpec:
    tag: str
    method_a: str
    method_b: str
    count: int


def _model_output_filename(method_name: str, condition: str) -> str:
    return f"{_slug(method_name)}_{_slug(condition)}.png"


def _load_meta_lookup(matrix_csv: Path) -> dict[str, dict]:
    if not matrix_csv.exists():
        raise FileNotFoundError(f"Matrix CSV not found: {matrix_csv}")

    df = pd.read_csv(matrix_csv)
    if df.empty:
        raise ValueError(f"Matrix CSV is empty: {matrix_csv}")
    df = df[df["condition"] == "Unconditional"].copy()
    if df.empty:
        raise ValueError("No unconditional rows found in matrix_results.csv.")
    df = df.groupby("sample_id", as_index=False)[["mask_type", "mask_coverage", "severity_bin"]].first()

    lookup: dict[str, dict] = {}
    for _, row in df.iterrows():
        sample_id = str(row["sample_id"])
        folder_id = sample_id[:-4] if sample_id.lower().endswith(".png") else sample_id
        coverage = float(row["mask_coverage"]) if pd.notna(row["mask_coverage"]) else 0.0
        lookup[folder_id] = {
            "mask_type": str(row.get("mask_type") or "unknown"),
            "mask_coverage": coverage,
            "coverage_bin": _coverage_bin(coverage),
            "severity_bin": str(row.get("severity_bin") or "unknown"),
        }
    return lookup


def _stratified_pick(
    sample_ids: list[str],
    *,
    meta_lookup: dict[str, dict],
    count: int,
    seed: int,
    salt: str,
) -> list[str]:
    if count <= 0:
        return []
    if len(sample_ids) < count:
        raise ValueError(f"Not enough candidates for '{salt}': need {count}, found {len(sample_ids)}.")

    rng = _rng(seed, salt)
    grouped: dict[tuple[str, str], list[str]] = {}
    for sid in sample_ids:
        meta = meta_lookup.get(sid)
        if meta is None:
            continue
        key = (meta["mask_type"], meta["coverage_bin"])
        grouped.setdefault(key, []).append(sid)

    selected: list[str] = []
    remaining = count

    for key in sorted(grouped.keys()):
        if remaining <= 0:
            break
        pool = grouped[key][:]
        rng.shuffle(pool)
        chosen = pool[0]
        if chosen not in selected:
            selected.append(chosen)
            remaining -= 1

    if remaining > 0:
        pool = sorted(set(sample_ids) - set(selected))
        rng.shuffle(pool)
        selected.extend(pool[:remaining])

    return selected


def _copy(path_from: Path, path_to: Path) -> None:
    path_to.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path_from, path_to)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build curated website evaluator pack from Stage 13 outputs.")
    parser.add_argument(
        "--stage13-root",
        default="outputs/13_model_evaluation",
        help="Path to Stage 13 artifact root.",
    )
    parser.add_argument(
        "--output-root",
        default="outputs/16_expert_validation/Website_Evaluator_Pack_v1",
        help="Output root for generated pack + ground truth.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Deterministic random seed.")
    parser.add_argument("--condition", default="Unconditional", help="Condition suffix to use for model outputs.")
    parser.add_argument("--blocka-count", type=int, default=13, help="Reserved item count for Block A source pool.")
    parser.add_argument(
        "--blocka-methods",
        nargs=2,
        default=["FT-SD", "Vanilla SD"],
        metavar=("METHOD_A", "METHOD_B"),
        help="Methods copied as A/B variants for Block A source items.",
    )
    parser.add_argument("--pair1-tag", default="PAIR_FTSD_TELEA")
    parser.add_argument("--pair1-a", default="Telea")
    parser.add_argument("--pair1-b", default="FT-SD")
    parser.add_argument("--pair1-count", type=int, default=24)
    parser.add_argument("--pair2-tag", default="PAIR_FTSD_LAMA")
    parser.add_argument("--pair2-a", default="LaMa")
    parser.add_argument("--pair2-b", default="FT-SD")
    parser.add_argument("--pair2-count", type=int, default=24)
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    stage13_root = Path(args.stage13_root)
    samples_dir = stage13_root / "samples"
    matrix_csv = stage13_root / "benchmarking_matrix" / "matrix_results.csv"
    if not samples_dir.exists():
        raise FileNotFoundError(f"Stage 13 samples directory not found: {samples_dir}")

    output_root = Path(args.output_root)
    pack_dir = output_root / "Expert_Pack_v2"
    gt_dir = output_root / "stage13_ground_truth"
    selection_csv = output_root / "selection_report.csv"
    summary_json = output_root / "pack_summary.json"
    import_guide = output_root / "IMPORT_INSTRUCTIONS.md"

    pair_specs = [
        PairSpec(args.pair1_tag, args.pair1_a, args.pair1_b, int(args.pair1_count)),
        PairSpec(args.pair2_tag, args.pair2_a, args.pair2_b, int(args.pair2_count)),
    ]
    blocka_methods = tuple(args.blocka_methods)
    condition = str(args.condition)
    seed = int(args.seed)
    blocka_count = int(args.blocka_count)

    meta_lookup = _load_meta_lookup(matrix_csv)

    needed_files = {"masked_input.png", "original.png"}
    needed_files.update({_model_output_filename(m, condition) for m in blocka_methods})
    for pair in pair_specs:
        needed_files.add(_model_output_filename(pair.method_a, condition))
        needed_files.add(_model_output_filename(pair.method_b, condition))

    valid_ids: list[str] = []
    missing_by_file: dict[str, int] = {name: 0 for name in sorted(needed_files)}
    for sample_dir in sorted([d for d in samples_dir.iterdir() if d.is_dir()]):
        sample_id = sample_dir.name
        if sample_id not in meta_lookup:
            continue
        missing = [f for f in needed_files if not (sample_dir / f).exists()]
        if missing:
            for fname in missing:
                missing_by_file[fname] += 1
            continue
        valid_ids.append(sample_id)

    if len(valid_ids) < (blocka_count + sum(p.count for p in pair_specs)):
        raise ValueError(
            "Not enough valid Stage 13 samples for requested pack size. "
            f"valid={len(valid_ids)} required={blocka_count + sum(p.count for p in pair_specs)}"
        )

    blocka_ids = _stratified_pick(
        valid_ids,
        meta_lookup=meta_lookup,
        count=blocka_count,
        seed=seed,
        salt="block-a",
    )
    remaining = sorted(set(valid_ids) - set(blocka_ids))

    pair_allocations: dict[str, list[str]] = {}
    for idx, pair in enumerate(pair_specs, start=1):
        picked = _stratified_pick(
            remaining,
            meta_lookup=meta_lookup,
            count=pair.count,
            seed=seed,
            salt=f"pair-{idx}-{pair.tag}",
        )
        pair_allocations[pair.tag] = picked
        remaining = sorted(set(remaining) - set(picked))

    if output_root.exists():
        shutil.rmtree(output_root)
    (pack_dir / "images").mkdir(parents=True, exist_ok=True)
    gt_dir.mkdir(parents=True, exist_ok=True)

    public_items: list[dict] = []
    private_items: list[dict] = []
    selection_rows: list[dict] = []

    blocka_a_file = _model_output_filename(blocka_methods[0], condition)
    blocka_b_file = _model_output_filename(blocka_methods[1], condition)
    for sid in blocka_ids:
        src = samples_dir / sid
        tagged_id = f"BLOCKA__{sid}"
        dst = pack_dir / "images" / tagged_id
        dst.mkdir(parents=True, exist_ok=True)
        _copy(src / "masked_input.png", dst / "input.png")
        _copy(src / blocka_a_file, dst / "A.png")
        _copy(src / blocka_b_file, dst / "B.png")
        _copy(src / "original.png", gt_dir / tagged_id / "ground_truth.png")

        meta = meta_lookup[sid]
        public_items.append(
            {
                "sample_id": tagged_id,
                "input": f"images/{tagged_id}/input.png",
                "A": f"images/{tagged_id}/A.png",
                "B": f"images/{tagged_id}/B.png",
                "pair_tag": "BLOCKA_ONLY",
                "pair_methods": [blocka_methods[0], blocka_methods[1]],
                **meta,
            }
        )
        private_items.append(
            {
                "sample_id": tagged_id,
                "original_sample_id": sid,
                "pair_tag": "BLOCKA_ONLY",
                "mapping": {"A": blocka_methods[0], "B": blocka_methods[1]},
            }
        )
        selection_rows.append({"bucket": "BLOCKA", "sample_id": sid, **meta})

    for pair in pair_specs:
        a_file = _model_output_filename(pair.method_a, condition)
        b_file = _model_output_filename(pair.method_b, condition)
        for sid in pair_allocations[pair.tag]:
            src = samples_dir / sid
            tagged_id = f"{pair.tag}__{sid}"
            dst = pack_dir / "images" / tagged_id
            dst.mkdir(parents=True, exist_ok=True)
            _copy(src / "masked_input.png", dst / "input.png")

            sid_hash = int(md5(tagged_id.encode("utf-8", errors="ignore")).hexdigest()[:8], 16)
            local_rng = random.Random(seed ^ sid_hash)
            options = [("A", pair.method_a, src / a_file), ("B", pair.method_b, src / b_file)]
            local_rng.shuffle(options)
            mapping = {}
            for label, method_name, source_path in options:
                _copy(source_path, dst / f"{label}.png")
                mapping[label] = method_name

            meta = meta_lookup[sid]
            public_items.append(
                {
                    "sample_id": tagged_id,
                    "input": f"images/{tagged_id}/input.png",
                    "A": f"images/{tagged_id}/A.png",
                    "B": f"images/{tagged_id}/B.png",
                    "pair_tag": pair.tag,
                    "pair_methods": [pair.method_a, pair.method_b],
                    **meta,
                }
            )
            private_items.append(
                {
                    "sample_id": tagged_id,
                    "original_sample_id": sid,
                    "pair_tag": pair.tag,
                    "mapping": mapping,
                }
            )
            selection_rows.append({"bucket": pair.tag, "sample_id": sid, **meta})

    created_at = datetime.now(timezone.utc).isoformat()
    manifest_public = {
        "pack_id": "Expert_Pack_v2",
        "created_at": created_at,
        "pairs": [
            {
                "tag": pair.tag,
                "method_pair": [pair.method_a, pair.method_b],
                "condition": condition,
                "count": pair.count,
            }
            for pair in pair_specs
        ],
        "block_a_pool": {
            "count": blocka_count,
            "methods": [blocka_methods[0], blocka_methods[1]],
            "disjoint_ordering_required": True,
            "reserved_first_n_items": blocka_count,
        },
        "rating_schema": {
            "task": "Pick the better restoration for the missing region (historical plausibility + visual coherence).",
            "fields": [
                {"name": "choice", "type": "enum", "values": ["A", "B", "Tie", "Unsure"]},
                {"name": "confidence", "type": "int", "min": 1, "max": 5},
                {"name": "comment", "type": "string", "optional": True},
            ],
        },
        "items": public_items,
    }
    manifest_private = {
        "pack_id": "Expert_Pack_v2",
        "created_at": created_at,
        "items": private_items,
    }

    (pack_dir / "manifest_public.json").write_text(json.dumps(manifest_public, indent=2), encoding="utf-8")
    (pack_dir / "manifest_private.json").write_text(json.dumps(manifest_private, indent=2), encoding="utf-8")
    pd.DataFrame(selection_rows).to_csv(selection_csv, index=False)

    block_b_total = sum(pair.count for pair in pair_specs)
    coverage_counts = (
        pd.DataFrame([r for r in selection_rows if r["bucket"] != "BLOCKA"])["coverage_bin"]
        .value_counts()
        .sort_index()
        .to_dict()
    )
    mask_counts = (
        pd.DataFrame([r for r in selection_rows if r["bucket"] != "BLOCKA"])["mask_type"]
        .value_counts()
        .sort_index()
        .to_dict()
    )
    summary = {
        "created_at": created_at,
        "seed": seed,
        "valid_candidates": len(valid_ids),
        "manifest_items_total": len(public_items),
        "block_a_reserved_items": blocka_count,
        "block_b_items_total": block_b_total,
        "ground_truth_samples": len(blocka_ids),
        "pair_allocations": {k: len(v) for k, v in pair_allocations.items()},
        "block_b_coverage_distribution": coverage_counts,
        "block_b_mask_type_distribution": mask_counts,
        "paths": {
            "pack_dir": str(pack_dir),
            "ground_truth_dir": str(gt_dir),
            "selection_report_csv": str(selection_csv),
        },
        "import_hint": (
            "cd website/services/api && "
            "python -m app.tools.import_pack "
            f"--pack-dir \"{pack_dir.as_posix()}\" "
            f"--stage13-samples \"{gt_dir.as_posix()}\" "
            "--campaign-name \"ATHENA Wave 1\" --seed 42"
        ),
    }
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    import_guide.write_text(
        "\n".join(
            [
                "# ATHENA Evaluator Pack Import Guide",
                "",
                "## Generated assets",
                f"- Pack directory: `{pack_dir.as_posix()}`",
                f"- Ground truth directory: `{gt_dir.as_posix()}`",
                f"- Selection report: `{selection_csv.as_posix()}`",
                "",
                "## Import command (from repository root)",
                "```powershell",
                "cd website/services/api",
                "python -m app.tools.import_pack "
                f"--pack-dir \"{pack_dir.as_posix()}\" "
                f"--stage13-samples \"{gt_dir.as_posix()}\" "
                "--campaign-name \"ATHENA Wave 1\" --seed 42",
                "```",
                "",
                "## Optional flags",
                "- Keep previous active campaign: add `--no-activate`",
                "- Allow Block A/Block B overlap: add `--allow-overlap`",
                "",
                "## Notes",
                f"- Manifest item ordering is intentional: first {blocka_count} items feed Block A generated pool when disjoint mode is enabled.",
                "- The remaining items feed Block B pairwise comparisons.",
                "",
            ]
        ),
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
