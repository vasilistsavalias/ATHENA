from __future__ import annotations

import argparse
import json
import random
import shutil
import unicodedata
import zipfile
from pathlib import Path


def _fold(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    return normalized.encode("ascii", "ignore").decode("ascii").lower()


def _resolve_sample_dir(samples_root: Path, sample_id: str) -> Path:
    candidates = [samples_root / sample_id, samples_root / Path(sample_id).stem]
    for candidate in candidates:
        if candidate.exists() and candidate.is_dir():
            return candidate

    target_variants = {sample_id, Path(sample_id).stem, _fold(sample_id), _fold(Path(sample_id).stem)}
    for child in samples_root.iterdir():
        if not child.is_dir():
            continue
        child_variants = {child.name, _fold(child.name)}
        if child_variants & target_variants:
            return child
    raise FileNotFoundError(f"Could not resolve sample directory for: {sample_id}")


def _copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _zip_dir(source_dir: Path, zip_path: Path) -> None:
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(source_dir.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(source_dir).as_posix())


def _item_paths(block: str, sample_id: str, include_cd: bool) -> dict[str, str]:
    payload = {
        "input": f"images/{block}/{sample_id}/input.png",
        "A": f"images/{block}/{sample_id}/A.png",
        "B": f"images/{block}/{sample_id}/B.png",
    }
    if include_cd:
        payload["C"] = f"images/{block}/{sample_id}/C.png"
        payload["D"] = f"images/{block}/{sample_id}/D.png"
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build 3-part expert pack zip with block_c_items.")
    parser.add_argument(
        "--base-pack-dir",
        default="outputs/S18_expert_validation/Website_Expert_Pack_Final_V8",
        help="Two-block base pack directory containing manifest_public/private and A/B images.",
    )
    parser.add_argument(
        "--samples-root",
        default="outputs/S15_model_evaluation/samples",
        help="Stage 15 samples root with model outputs.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/S18_expert_validation/Website_Expert_Pack_Final_V8_3part",
        help="Output directory for the generated 3-part pack.",
    )
    parser.add_argument(
        "--output-zip",
        default="outputs/S18_expert_validation/Website_Expert_Pack_Final_V8_3part.zip",
        help="Output zip path.",
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--block-a-target", type=int, default=10)
    parser.add_argument("--block-b-target", type=int, default=28)
    parser.add_argument("--block-c-count", type=int, default=16)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    base_pack_dir = Path(args.base_pack_dir)
    samples_root = Path(args.samples_root)
    output_dir = Path(args.output_dir)
    output_zip = Path(args.output_zip)

    manifest_public = json.loads((base_pack_dir / "manifest_public.json").read_text(encoding="utf-8"))
    manifest_private = json.loads((base_pack_dir / "manifest_private.json").read_text(encoding="utf-8"))
    practice_items = list(manifest_public.get("practice_items", []))
    scored_items = list(manifest_public.get("items", []))
    private_items = {str(item.get("sample_id")): item for item in manifest_private.get("items", [])}
    private_practice = {str(item.get("sample_id")): item for item in manifest_private.get("practice_items", [])}

    if len(scored_items) < int(args.block_c_count):
        raise ValueError("Not enough scored items in base pack to derive block_c_items.")

    rng = random.Random(int(args.seed))
    block_c_candidates: list[dict] = []
    for src_item in scored_items:
        sample_id = str(src_item["sample_id"])
        try:
            sample_dir = _resolve_sample_dir(samples_root, sample_id)
            required = [
                "FT_SD_TTA_Unconditional.png",
                "MAT_Unconditional.png",
                "CoModGAN_Unconditional.png",
                "LaMa_Unconditional.png",
                "masked_input.png",
            ]
            if all((sample_dir / filename).exists() for filename in required):
                block_c_candidates.append(src_item)
        except FileNotFoundError:
            continue
    rng.shuffle(block_c_candidates)
    block_c_source = block_c_candidates[: int(args.block_c_count)]
    if len(block_c_source) < int(args.block_c_count):
        raise ValueError(
            f"Could not build Block C with requested count={int(args.block_c_count)}. "
            f"Only {len(block_c_source)} eligible scored samples were found."
        )

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Copy existing block B assets from base pack into new block_b folder.
    for section_name, items in (("practice_items", practice_items), ("items", scored_items)):
        for item in items:
            sample_id = str(item["sample_id"])
            source_sample_dir = base_pack_dir / "images" / "block_b" / sample_id
            if not source_sample_dir.exists():
                source_sample_dir = base_pack_dir / "images" / sample_id
            if not source_sample_dir.exists():
                raise FileNotFoundError(f"Missing base image directory for sample_id={sample_id}")
            _copy(source_sample_dir / "input.png", output_dir / "images" / "block_b" / sample_id / "input.png")
            _copy(source_sample_dir / "A.png", output_dir / "images" / "block_b" / sample_id / "A.png")
            _copy(source_sample_dir / "B.png", output_dir / "images" / "block_b" / sample_id / "B.png")
            item.update(_item_paths("block_b", sample_id, include_cd=False))

    block_c_items: list[dict] = []
    private_block_c_items: list[dict] = []
    models = [
        ("FT-SD+TTA", "FT_SD_TTA_Unconditional.png"),
        ("MAT", "MAT_Unconditional.png"),
        ("CoModGAN", "CoModGAN_Unconditional.png"),
        ("LaMa", "LaMa_Unconditional.png"),
    ]
    labels = ["A", "B", "C", "D"]
    for src_item in block_c_source:
        sample_id = str(src_item["sample_id"])
        sample_dir = _resolve_sample_dir(samples_root, sample_id)
        for _, filename in models:
            if not (sample_dir / filename).exists():
                raise FileNotFoundError(f"Missing model output {filename} for sample_id={sample_id}")

        order = list(models)
        rng.shuffle(order)
        mapping: dict[str, str] = {}
        output_sample_dir = output_dir / "images" / "block_c" / sample_id
        _copy(sample_dir / "masked_input.png", output_sample_dir / "input.png")
        for label, (model_name, filename) in zip(labels, order):
            _copy(sample_dir / filename, output_sample_dir / f"{label}.png")
            mapping[label] = model_name

        block_c_item = {
            "sample_id": sample_id,
            "mask_type": src_item.get("mask_type"),
            "coverage_bin": src_item.get("coverage_bin"),
            "mask_coverage": src_item.get("mask_coverage"),
            "is_anchor": bool(src_item.get("is_anchor", False)),
            "is_practice": False,
            "block_part": "C",
            **_item_paths("block_c", sample_id, include_cd=True),
        }
        block_c_items.append(block_c_item)
        private_block_c_items.append(
            {
                "sample_id": sample_id,
                "is_practice": False,
                "is_anchor": bool(src_item.get("is_anchor", False)),
                "mapping": mapping,
            }
        )

    updated_public = {
        **manifest_public,
        "study_mode": "two_block",
        "block_a_target_count": int(args.block_a_target),
        "block_b_target_count": int(args.block_b_target),
        "practice_items": [{**item, "is_practice": True} for item in practice_items],
        "items": [{**item, "is_practice": False, "block_part": "B"} for item in scored_items],
        "block_c_items": block_c_items,
    }
    updated_private = {
        **manifest_private,
        "practice_items": [private_practice.get(str(item["sample_id"]), {"sample_id": str(item["sample_id"]), "mapping": {}}) for item in practice_items],
        "items": [private_items.get(str(item["sample_id"]), {"sample_id": str(item["sample_id"]), "mapping": {}}) for item in scored_items],
        "block_c_items": private_block_c_items,
    }

    (output_dir / "manifest_public.json").write_text(json.dumps(updated_public, indent=2), encoding="utf-8")
    (output_dir / "manifest_private.json").write_text(json.dumps(updated_private, indent=2), encoding="utf-8")
    (output_dir / "README_IMPORT.txt").write_text(
        "Three-part expert pack with block_c_items for website upload/import.\n",
        encoding="utf-8",
    )

    if output_zip.exists():
        output_zip.unlink()
    _zip_dir(output_dir, output_zip)

    summary = {
        "output_dir": str(output_dir),
        "output_zip": str(output_zip),
        "practice_items": len(practice_items),
        "block_b_items": len(scored_items),
        "block_c_items": len(block_c_items),
        "block_a_target_count": int(args.block_a_target),
        "block_b_target_count": int(args.block_b_target),
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
