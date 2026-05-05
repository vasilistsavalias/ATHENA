from __future__ import annotations

import json
import math
import shutil
import unicodedata
from pathlib import Path
import re

from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.models import BlockAItem, BlockBItem, Campaign

DISALLOWED_COVERAGE_BINS = {">50%"}


def _safe_coverage_bin(value):
    if value is None:
        return None
    return str(value)


def _validate_pack_coverage_bins(items: list[dict]) -> None:
    violations: list[str] = []
    for item in items:
        coverage_bin = _safe_coverage_bin(item.get("coverage_bin"))
        if coverage_bin in DISALLOWED_COVERAGE_BINS:
            violations.append(str(item.get("sample_id", "<unknown>")))
    if violations:
        preview = ", ".join(violations[:5])
        suffix = " ..." if len(violations) > 5 else ""
        raise ValueError(
            "Pack import blocked: found out-of-policy mask coverage bins (>50%). "
            f"Affected sample_ids: {preview}{suffix}"
        )


def _normalize_web_path(path: str) -> str:
    return path.replace("\\", "/")


def _safe_segment(value: str) -> str:
    text = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("._")
    return text or "item"


def _sanitize_relative_path(relative_path: str) -> str:
    parts = [part for part in Path(relative_path).parts if part not in {"", ".", ".."}]
    if not parts:
        raise ValueError("Invalid relative path.")
    return _normalize_web_path("/".join(_safe_segment(part) for part in parts))


def _copy_into_campaign_storage(
    *,
    campaign_storage_root: Path,
    source_root: Path,
    relative_path: str,
) -> str:
    source = source_root / relative_path
    if not source.exists():
        raise FileNotFoundError(f"Missing file in pack: {source}")
    sanitized_relative_path = _sanitize_relative_path(relative_path)
    destination = campaign_storage_root / sanitized_relative_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    normalized = _normalize_web_path(sanitized_relative_path)
    return f"/static/{campaign_storage_root.name}/{normalized}"


def _load_private_mapping(pack_dir: Path) -> dict[str, dict]:
    private_manifest = pack_dir / "manifest_private.json"
    if not private_manifest.exists():
        return {}
    payload = json.loads(private_manifest.read_text(encoding="utf-8"))
    mapping_by_sample_id: dict[str, dict] = {}
    for item in [*payload.get("practice_items", []), *payload.get("items", [])]:
        sample_id = str(item.get("sample_id", "")).strip()
        mapping = item.get("mapping") or {}
        if sample_id and isinstance(mapping, dict):
            mapping_by_sample_id[sample_id] = mapping
    return mapping_by_sample_id


def _resolve_candidate_roles(mapping: dict) -> tuple[str | None, str | None]:
    calibration_choice = None
    restoration_choice = None
    for choice in ("A", "B"):
        value = str(mapping.get(choice, "")).strip()
        if not value:
            continue
        if value == "Ground-Truth":
            calibration_choice = choice
        else:
            restoration_choice = choice
    return calibration_choice, restoration_choice


def import_pack(
    db: Session,
    *,
    pack_dir: Path,
    campaign_name: str,
    seed: int,
    stage13_samples: Path | None = None,
    activate: bool = True,
    disjoint_blocks: bool = True,
) -> Campaign:
    settings = get_settings()
    manifest_path = pack_dir / "manifest_public.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Could not find manifest_public.json in {pack_dir}")

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    private_mapping_by_sample_id = _load_private_mapping(pack_dir)
    scored_items = payload.get("items", [])
    practice_items = payload.get("practice_items", [])
    block_c_items = payload.get("block_c_items", [])
    explicit_block_a_items = payload.get("block_a_items", [])
    study_mode = str(payload.get("study_mode") or "two_block")
    all_block_b_items = [*practice_items, *scored_items, *block_c_items]
    if not all_block_b_items:
        raise ValueError("manifest_public.json contains zero items.")
    _validate_pack_coverage_bins(all_block_b_items)

    is_pairwise_only = study_mode == "pairwise_only"
    manifest_block_a_target = int(payload.get("block_a_target_count", 0 if is_pairwise_only else settings.block_a_target_count) or 0)
    default_block_b_target = len(all_block_b_items) if is_pairwise_only else settings.block_b_target_count
    manifest_block_b_target = int(payload.get("block_b_target_count", default_block_b_target) or 0)
    if block_c_items:
        # For three-part studies, Block B target stores total pairwise rows (B + C) for compatibility.
        if "block_b_target_count" not in payload:
            manifest_block_b_target = len(all_block_b_items)

    if activate:
        db.query(Campaign).update({Campaign.is_active: False})

    campaign = Campaign(
        name=campaign_name,
        seed=int(seed),
        is_active=bool(activate),
        protocol_version=settings.expert_protocol_version,
        block_a_target_count=manifest_block_a_target,
        block_b_target_count=manifest_block_b_target,
    )
    db.add(campaign)
    db.flush()

    campaign_storage = settings.storage_root / "campaigns" / f"{campaign.id}"
    campaign_storage.mkdir(parents=True, exist_ok=True)

    real_candidates: list[tuple[str, Path]] = []
    if stage13_samples is not None and stage13_samples.exists():
        for sample_dir in sorted([p for p in stage13_samples.iterdir() if p.is_dir()]):
            gt = sample_dir / "ground_truth.png"
            if gt.exists():
                real_candidates.append((sample_dir.name, gt))

    # Curated packs may provide an explicit Block A pool. In that case Block A does
    # not need to reserve any scored pairwise samples from Block B.
    uses_explicit_block_a = bool(explicit_block_a_items)
    reserve_for_block_a = (
        0
        if uses_explicit_block_a
        else math.ceil(manifest_block_a_target / 2) if disjoint_blocks and manifest_block_a_target > 0 else 0
    )

    pairwise_target_b_only = max(0, manifest_block_b_target - len(block_c_items))
    if not is_pairwise_only and disjoint_blocks:
        max_reservable = max(0, len(scored_items) - pairwise_target_b_only)
        if reserve_for_block_a > max_reservable:
            raise ValueError(
                "Cannot keep Block A and Block B disjoint with current pack size. "
                f"Needed reserved={reserve_for_block_a}, max reservable={max_reservable}."
            )
        block_a_source_items = [] if uses_explicit_block_a else scored_items[:reserve_for_block_a]
        block_b_source_items = scored_items[reserve_for_block_a:]
    else:
        block_a_source_items = scored_items
        block_b_source_items = all_block_b_items if is_pairwise_only else scored_items

    block_c_sample_ids = {str(item.get("sample_id", "")) for item in block_c_items}
    block_b_sample_ids: set[str] = set()
    block_b_rows: list[BlockBItem] = []
    pairwise_source_items = [*practice_items, *block_b_source_items, *block_c_items]
    for item in pairwise_source_items:
        sample_id = str(item["sample_id"])
        private_mapping = private_mapping_by_sample_id.get(sample_id, {})
        calibration_choice, restoration_choice = _resolve_candidate_roles(private_mapping)
        block_b_sample_ids.add(sample_id)
        metadata_json = {
            "pack_sample_id": sample_id,
            "mask_coverage": item.get("mask_coverage"),
            "disjoint_blocks": disjoint_blocks,
            "is_practice": bool(item.get("is_practice", False)),
            "is_anchor": bool(item.get("is_anchor", False)),
            "study_mode": study_mode,
            "protocol_version": campaign.protocol_version,
            "candidate_mapping": private_mapping,
            "calibration_choice": calibration_choice,
            "restoration_choice": restoration_choice,
            "block_part": str(item.get("block_part") or ("C" if sample_id in block_c_sample_ids else "B")).upper(),
        }
        option_c_relative = item.get("C")
        option_d_relative = item.get("D")
        if option_c_relative and option_d_relative:
            metadata_json["option_c_url"] = _copy_into_campaign_storage(
                campaign_storage_root=campaign_storage,
                source_root=pack_dir,
                relative_path=str(option_c_relative),
            )
            metadata_json["option_d_url"] = _copy_into_campaign_storage(
                campaign_storage_root=campaign_storage,
                source_root=pack_dir,
                relative_path=str(option_d_relative),
            )
        block_b_rows.append(
            BlockBItem(
                campaign_id=campaign.id,
                sample_id=sample_id,
                input_url=_copy_into_campaign_storage(
                    campaign_storage_root=campaign_storage,
                    source_root=pack_dir,
                    relative_path=str(item["input"]),
                ),
                option_a_url=_copy_into_campaign_storage(
                    campaign_storage_root=campaign_storage,
                    source_root=pack_dir,
                    relative_path=str(item["A"]),
                ),
                option_b_url=_copy_into_campaign_storage(
                    campaign_storage_root=campaign_storage,
                    source_root=pack_dir,
                    relative_path=str(item["B"]),
                ),
                mask_type=item.get("mask_type"),
                mask_coverage_bin=_safe_coverage_bin(item.get("coverage_bin")),
                metadata_json=metadata_json,
            )
        )

    db.add_all(block_b_rows)
    db.flush()

    block_a_rows: list[BlockAItem] = []
    if not is_pairwise_only and manifest_block_a_target > 0:
        if explicit_block_a_items:
            for index, item in enumerate(explicit_block_a_items, start=1):
                sample_id = str(item.get("sample_id") or f"explicit_block_a_{index}")
                image_relative = item.get("image")
                if not image_relative:
                    raise ValueError(f"block_a_items[{index - 1}] is missing required 'image' path.")
                image_url = _copy_into_campaign_storage(
                    campaign_storage_root=campaign_storage,
                    source_root=pack_dir,
                    relative_path=str(image_relative),
                )
                metadata_json = dict(item.get("metadata_json") or {})
                metadata_json.setdefault("origin", "manifest_block_a_items")
                block_a_rows.append(
                    BlockAItem(
                        campaign_id=campaign.id,
                        sample_id=sample_id,
                        image_url=image_url,
                        source_label=str(item.get("source_label") or "generated"),
                        mask_type=item.get("mask_type"),
                        mask_coverage_bin=_safe_coverage_bin(item.get("coverage_bin")),
                        metadata_json=metadata_json,
                    )
                )
        else:
            for sid, gt_path in real_candidates:
                if disjoint_blocks and sid in block_b_sample_ids:
                    continue
                rel_path = f"block_a_real/{sid}.png"
                final_path = campaign_storage / rel_path
                final_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(gt_path, final_path)
                gt_url = f"/static/{campaign_storage.name}/{_normalize_web_path(rel_path)}"
                block_a_rows.append(
                    BlockAItem(
                        campaign_id=campaign.id,
                        sample_id=sid,
                        image_url=gt_url,
                        source_label="real",
                        mask_type=None,
                        mask_coverage_bin=None,
                        metadata_json={"origin": "stage13_ground_truth"},
                    )
                )

            source_items_for_generated_a = block_a_source_items if disjoint_blocks else block_b_source_items

            for item in source_items_for_generated_a:
                sample_id = str(item["sample_id"])
                option_a_url = _copy_into_campaign_storage(
                    campaign_storage_root=campaign_storage,
                    source_root=pack_dir,
                    relative_path=str(item["A"]),
                )
                option_b_url = _copy_into_campaign_storage(
                    campaign_storage_root=campaign_storage,
                    source_root=pack_dir,
                    relative_path=str(item["B"]),
                )
                mask_type = item.get("mask_type")
                coverage_bin = _safe_coverage_bin(item.get("coverage_bin"))
                block_a_rows.append(
                    BlockAItem(
                        campaign_id=campaign.id,
                        sample_id=f"blkA_{sample_id}_A",
                        image_url=option_a_url,
                        source_label="generated",
                        mask_type=mask_type,
                        mask_coverage_bin=coverage_bin,
                        metadata_json={"origin_sample_id": sample_id, "origin_variant": "A"},
                    )
                )
                block_a_rows.append(
                    BlockAItem(
                        campaign_id=campaign.id,
                        sample_id=f"blkA_{sample_id}_B",
                        image_url=option_b_url,
                        source_label="generated",
                        mask_type=mask_type,
                        mask_coverage_bin=coverage_bin,
                        metadata_json={"origin_sample_id": sample_id, "origin_variant": "B"},
                    )
                )

    if manifest_block_a_target > 0 and len(block_a_rows) < manifest_block_a_target:
        raise ValueError(
            f"Imported Block A pool has only {len(block_a_rows)} items; "
            f"requires at least {manifest_block_a_target}."
        )
    if len(block_b_rows) < manifest_block_b_target:
        raise ValueError(
            f"Imported Block B pool has only {len(block_b_rows)} items; "
            f"requires at least {manifest_block_b_target}."
        )

    if block_a_rows:
        db.add_all(block_a_rows)
    db.commit()
    db.refresh(campaign)
    return campaign
