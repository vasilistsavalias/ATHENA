"""Convert the final Stage 18 pack into a website-ready pairwise-only campaign pack.

This keeps the Stage 18 sample selection and A/B blinding, but makes the pack
compatible with the ATHENA website flow:

- adds `input` images from Stage 15 `masked_input.png`
- splits out 2-3 practice trials (excluded from analysis)
- reduces anchor density to a smaller scored subset
- emits a pairwise-only manifest that the website can ingest directly
"""

from __future__ import annotations

import argparse
import json
import random
import re
import shutil
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import md5
from pathlib import Path


def _rng(seed: int, salt: str) -> random.Random:
    digest = md5(f"{seed}:{salt}".encode("utf-8")).hexdigest()[:8]
    return random.Random(int(digest, 16) ^ int(seed))


def _safe_float(value) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


@dataclass(frozen=True)
class PackItem:
    sample_id: str
    input_rel: str | None
    a_rel: str
    b_rel: str
    mask_type: str | None
    coverage_bin: str | None
    mask_coverage: float
    is_anchor: bool

    @classmethod
    def from_public(cls, item: dict) -> "PackItem":
        return cls(
            sample_id=str(item["sample_id"]),
            input_rel=str(item["input"]) if item.get("input") else None,
            a_rel=str(item["A"]),
            b_rel=str(item["B"]),
            mask_type=item.get("mask_type"),
            coverage_bin=item.get("coverage_bin"),
            mask_coverage=_safe_float(item.get("mask_coverage")),
            is_anchor=bool(item.get("is_anchor", False)),
        )

    def to_public_dict(self) -> dict:
        return {
            "sample_id": self.sample_id,
            "input": self.input_rel,
            "A": self.a_rel,
            "B": self.b_rel,
            "mask_type": self.mask_type,
            "coverage_bin": self.coverage_bin,
            "mask_coverage": self.mask_coverage,
            "is_anchor": self.is_anchor,
        }


def _pick_practice_items(items: list[PackItem], practice_count: int, seed: int) -> tuple[list[PackItem], list[PackItem]]:
    pool = [item for item in items if not item.is_anchor]
    rng = _rng(seed, "practice")
    by_group: dict[tuple[str | None, str | None], list[PackItem]] = {}
    for item in pool:
        by_group.setdefault((item.mask_type, item.coverage_bin), []).append(item)
    selected: list[PackItem] = []
    for group_key in sorted(by_group.keys(), key=lambda x: (str(x[0]), str(x[1]))):
        candidates = list(by_group[group_key])
        rng.shuffle(candidates)
        if candidates:
            selected.append(candidates[0])
        if len(selected) >= practice_count:
            break
    if len(selected) < practice_count:
        remaining = [item for item in pool if item.sample_id not in {x.sample_id for x in selected}]
        rng.shuffle(remaining)
        selected.extend(remaining[: practice_count - len(selected)])
    practice_ids = {item.sample_id for item in selected[:practice_count]}
    return selected[:practice_count], [item for item in items if item.sample_id not in practice_ids]


def _pick_anchor_subset(items: list[PackItem], anchor_count: int) -> set[str]:
    anchors = sorted([item for item in items if item.is_anchor], key=lambda x: (x.mask_coverage, x.sample_id))
    if anchor_count <= 0 or not anchors:
        return set()
    if len(anchors) <= anchor_count:
        return {item.sample_id for item in anchors}

    indices: list[int] = []
    if anchor_count == 1:
        indices = [len(anchors) // 2]
    else:
        last_index = len(anchors) - 1
        for step in range(anchor_count):
            indices.append(round((last_index * step) / (anchor_count - 1)))
    return {anchors[index].sample_id for index in sorted(set(indices))}


def _copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _resolve_stage15_sample_dir(samples_root: Path, sample_id: str) -> Path | None:
    def _fold(value: str) -> str:
        normalized = unicodedata.normalize("NFKD", value)
        ascii_only = normalized.encode("ascii", "ignore").decode("ascii")
        return ascii_only.lower()

    candidates = [samples_root / sample_id, samples_root / Path(sample_id).stem]
    for candidate in candidates:
        if candidate.exists() and candidate.is_dir():
            return candidate
    target_variants = {
        sample_id,
        Path(sample_id).stem,
        unicodedata.normalize("NFC", sample_id),
        unicodedata.normalize("NFC", Path(sample_id).stem),
        unicodedata.normalize("NFKD", sample_id),
        unicodedata.normalize("NFKD", Path(sample_id).stem),
        _fold(sample_id),
        _fold(Path(sample_id).stem),
    }
    for child in samples_root.iterdir():
        if not child.is_dir():
            continue
        child_variants = {
            child.name,
            unicodedata.normalize("NFC", child.name),
            unicodedata.normalize("NFKD", child.name),
            _fold(child.name),
        }
        if child_variants & target_variants:
            return child
    stem = Path(sample_id).stem
    prefix_match = re.match(r"^((?:wiki|eur)_[^_]+)", stem)
    suffix_match = re.search(r"(_crop\d+_m\d+|_m\d+)$", stem)
    if prefix_match and suffix_match:
        prefix = prefix_match.group(1)
        sample_suffix = suffix_match.group(1)
        candidates = [
            child for child in samples_root.iterdir()
            if child.is_dir()
            and child.name.startswith(prefix)
            and (child.name.endswith(sample_suffix) or child.name.endswith(f"{sample_suffix}.png"))
        ]
        if len(candidates) == 1:
            return candidates[0]
    return None


def _resolve_pack_asset(pack_root: Path, relative_path: str) -> Path:
    candidate = pack_root / relative_path
    if candidate.exists():
        return candidate
    path_obj = Path(relative_path)
    if len(path_obj.parts) >= 3:
        sample_id = path_obj.parts[-2]
        filename = path_obj.parts[-1]
        images_root = pack_root / "images"
        resolved_dir = _resolve_stage15_sample_dir(images_root, sample_id)
        if resolved_dir is not None:
            fallback = resolved_dir / filename
            if fallback.exists():
                return fallback
    raise FileNotFoundError(f"Missing asset in pack: {pack_root / relative_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build website-ready pack from final Stage 18 outputs.")
    parser.add_argument(
        "--stage18-pack",
        default="outputs/S18_expert_validation/Expert_Pack_Top1vsReal",
        help="Path to final Stage 18 public/private pack directory.",
    )
    parser.add_argument(
        "--stage15-samples",
        default="outputs/S15_model_evaluation/samples",
        help="Path to Stage 15 sample folders containing masked_input.png.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/S18_expert_validation/Website_Expert_Pack_Final_V8",
        help="Output directory for the converted website pack.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--practice-count", type=int, default=3)
    parser.add_argument("--anchor-count", type=int, default=3)
    parser.add_argument("--participant-count", type=int, default=10)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    stage18_pack = Path(args.stage18_pack)
    stage15_samples = Path(args.stage15_samples)
    output_dir = Path(args.output_dir)

    manifest_public_path = stage18_pack / "manifest_public.json"
    manifest_private_path = stage18_pack / "manifest_private.json"
    if not manifest_public_path.exists() or not manifest_private_path.exists():
        raise FileNotFoundError("Stage 18 pack must contain manifest_public.json and manifest_private.json.")

    payload_public = _load_json(manifest_public_path)
    payload_private = _load_json(manifest_private_path)

    public_items = [PackItem.from_public(item) for item in payload_public.get("items", [])]
    if not public_items:
        raise ValueError("Stage 18 public manifest contains zero items.")

    private_lookup = {str(item["sample_id"]): item for item in payload_private.get("items", [])}
    practice_items, scored_pool = _pick_practice_items(public_items, int(args.practice_count), int(args.seed))
    retained_anchor_ids = _pick_anchor_subset(scored_pool, int(args.anchor_count))

    normalized_scored: list[PackItem] = []
    for item in scored_pool:
        normalized_scored.append(
            PackItem(
                sample_id=item.sample_id,
                input_rel=item.input_rel,
                a_rel=item.a_rel,
                b_rel=item.b_rel,
                mask_type=item.mask_type,
                coverage_bin=item.coverage_bin,
                mask_coverage=item.mask_coverage,
                is_anchor=item.sample_id in retained_anchor_ids,
            )
        )

    if output_dir.exists():
        shutil.rmtree(output_dir)
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    def materialize(item: PackItem) -> dict:
        sample_dir = _resolve_stage15_sample_dir(stage15_samples, item.sample_id)
        if sample_dir is None:
            raise FileNotFoundError(f"Missing Stage 15 sample directory for sample_id={item.sample_id}")
        input_path = sample_dir / "masked_input.png"
        if not input_path.exists():
            raise FileNotFoundError(f"Missing Stage 15 masked_input.png for sample_id={item.sample_id}: {input_path}")

        dest_dir = images_dir / item.sample_id
        _copy(input_path, dest_dir / "input.png")
        _copy(_resolve_pack_asset(stage18_pack, item.a_rel), dest_dir / "A.png")
        _copy(_resolve_pack_asset(stage18_pack, item.b_rel), dest_dir / "B.png")
        return PackItem(
            sample_id=item.sample_id,
            input_rel=f"images/{item.sample_id}/input.png",
            a_rel=f"images/{item.sample_id}/A.png",
            b_rel=f"images/{item.sample_id}/B.png",
            mask_type=item.mask_type,
            coverage_bin=item.coverage_bin,
            mask_coverage=item.mask_coverage,
            is_anchor=item.is_anchor,
        ).to_public_dict()

    public_practice = [materialize(item) for item in practice_items]
    public_scored = [materialize(item) for item in normalized_scored]

    private_practice = [private_lookup[item["sample_id"]] for item in public_practice]
    private_scored = [private_lookup[item["sample_id"]] for item in public_scored]

    created_at = datetime.now(timezone.utc).isoformat()
    prompt = (
        "Given the damaged input shown above, which candidate is the more plausible "
        "restoration of the missing region?"
    )
    manifest_public = {
        "pack_id": "Website_Expert_Pack_Final_V8",
        "created_at": created_at,
        "study_mode": "pairwise_only",
        "source_pack_id": payload_public.get("pack_id"),
        "participant_count": int(args.participant_count),
        "block_a_target_count": 0,
        "block_b_target_count": len(public_practice) + len(public_scored),
        "session_structure": {
            "practice_count": len(public_practice),
            "scored_count": len(public_scored),
            "anchor_count": sum(1 for item in public_scored if item["is_anchor"]),
        },
        "rating_schema": {
            "task": prompt,
            "fields": [
                {"name": "choice", "type": "enum", "values": ["A", "B", "Tie", "Unsure"]},
                {"name": "confidence", "type": "int", "min": 1, "max": 5},
                {"name": "comment", "type": "string", "optional": True},
            ],
        },
        "practice_items": [{**item, "is_practice": True} for item in public_practice],
        "items": [{**item, "is_practice": False} for item in public_scored],
        "analysis_exports": {
            "group_by": ["mask_type", "coverage_bin", "is_anchor", "is_practice"],
            "exclude_from_headline_metrics": ["is_practice", "is_anchor"],
        },
    }
    manifest_private = {
        "pack_id": "Website_Expert_Pack_Final_V8",
        "created_at": created_at,
        "study_mode": "pairwise_only",
        "practice_items": [{**item, "is_practice": True} for item in private_practice],
        "items": [{**item, "is_practice": False, "is_anchor": item["sample_id"] in retained_anchor_ids} for item in private_scored],
    }

    (output_dir / "manifest_public.json").write_text(json.dumps(manifest_public, indent=2), encoding="utf-8")
    (output_dir / "manifest_private.json").write_text(json.dumps(manifest_private, indent=2), encoding="utf-8")
    (output_dir / "README_IMPORT.txt").write_text(
        "\n".join(
            [
                "Stage 18 website-ready expert pack",
                "",
                "Import command:",
                "cd website/services/api",
                f"python -m app.tools.import_pack --pack-dir \"{output_dir.as_posix()}\" --campaign-name \"ATHENA Final V8 Expert Validation\" --seed {int(args.seed)}",
            ]
        ),
        encoding="utf-8",
    )

    summary = {
        "source_pack": str(stage18_pack),
        "output_dir": str(output_dir),
        "participant_count": int(args.participant_count),
        "practice_count": len(public_practice),
        "scored_count": len(public_scored),
        "anchor_count": sum(1 for item in public_scored if item["is_anchor"]),
        "sample_ids_practice": [item["sample_id"] for item in public_practice],
        "sample_ids_scored": [item["sample_id"] for item in public_scored],
    }
    (output_dir / "conversion_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
