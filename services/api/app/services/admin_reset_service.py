from __future__ import annotations

import shutil
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import func, text
from sqlalchemy.orm import Session

from app.models import (
    AttentionFlag,
    AuditEvent,
    BlockAAssignment,
    BlockAResponse,
    BlockBAssignment,
    BlockBResponse,
    Campaign,
    Participant,
    StageFeedback,
)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _resolve_target_campaign_ids(
    db: Session,
    *,
    campaign_id: int | None,
    all_active_campaigns: bool,
    all_campaigns: bool,
) -> list[int]:
    if campaign_id is not None:
        campaign = db.query(Campaign).filter(Campaign.id == campaign_id).first()
        if campaign is None:
            raise ValueError(f"Campaign not found: {campaign_id}")
        return [int(campaign.id)]

    if all_campaigns:
        campaign_ids = [int(row[0]) for row in db.query(Campaign.id).order_by(Campaign.id.asc()).all()]
        if not campaign_ids:
            raise ValueError("No campaign found.")
        return campaign_ids

    if all_active_campaigns:
        active_ids = [
            int(row[0])
            for row in db.query(Campaign.id)
            .filter(Campaign.is_active.is_(True))
            .order_by(Campaign.id.asc())
            .all()
        ]
        if not active_ids:
            raise ValueError("No active campaign found.")
        return active_ids

    active_campaign = db.query(Campaign).filter(Campaign.is_active.is_(True)).order_by(Campaign.id.desc()).first()
    if active_campaign is None:
        raise ValueError("No active campaign found.")
    return [int(active_campaign.id)]


def _delete_runtime_rows(db: Session, *, campaign_ids: list[int]) -> tuple[dict[str, int], list[int]]:
    participant_ids = [
        int(row[0])
        for row in db.query(Participant.id)
        .filter(Participant.campaign_id.in_(campaign_ids))
        .all()
    ]
    deleted_counts: dict[str, int] = {}
    deleted_counts["block_a_responses"] = (
        db.query(BlockAResponse).filter(BlockAResponse.campaign_id.in_(campaign_ids)).delete(synchronize_session=False)
    )
    deleted_counts["block_b_responses"] = (
        db.query(BlockBResponse).filter(BlockBResponse.campaign_id.in_(campaign_ids)).delete(synchronize_session=False)
    )
    deleted_counts["block_a_assignments"] = (
        db.query(BlockAAssignment).filter(BlockAAssignment.campaign_id.in_(campaign_ids)).delete(synchronize_session=False)
    )
    deleted_counts["block_b_assignments"] = (
        db.query(BlockBAssignment).filter(BlockBAssignment.campaign_id.in_(campaign_ids)).delete(synchronize_session=False)
    )
    deleted_counts["stage_feedback"] = (
        db.query(StageFeedback).filter(StageFeedback.campaign_id.in_(campaign_ids)).delete(synchronize_session=False)
    )
    deleted_counts["attention_flags"] = (
        db.query(AttentionFlag).filter(AttentionFlag.campaign_id.in_(campaign_ids)).delete(synchronize_session=False)
    )
    if participant_ids:
        deleted_counts["audit_events"] = (
            db.query(AuditEvent)
            .filter(
                (AuditEvent.campaign_id.in_(campaign_ids))
                | (AuditEvent.participant_id.in_(participant_ids))
            )
            .delete(synchronize_session=False)
        )
    else:
        deleted_counts["audit_events"] = (
            db.query(AuditEvent).filter(AuditEvent.campaign_id.in_(campaign_ids)).delete(synchronize_session=False)
        )
    deleted_counts["participants"] = (
        db.query(Participant).filter(Participant.campaign_id.in_(campaign_ids)).delete(synchronize_session=False)
    )
    return deleted_counts, participant_ids


def _remove_campaign_asset_dirs(storage_root: Path, campaign_ids: list[int]) -> list[str]:
    removed: list[str] = []
    campaigns_root = storage_root / "campaigns"
    for campaign_id in campaign_ids:
        target = campaigns_root / str(campaign_id)
        if target.exists():
            shutil.rmtree(target)
            removed.append(str(target))
    return removed


def _reset_participant_identity_if_empty(db: Session) -> bool:
    participants_remaining = int(db.query(Participant.id).count())
    if participants_remaining != 0:
        return False

    bind = db.get_bind()
    dialect = bind.dialect.name if bind is not None else ""
    if dialect == "sqlite":
        has_sequence_table = db.execute(
            text("SELECT name FROM sqlite_master WHERE type='table' AND name='sqlite_sequence'")
        ).first()
        if has_sequence_table:
            db.execute(text("DELETE FROM sqlite_sequence WHERE name = 'participants'"))
        return True
    if dialect in {"postgresql", "postgres"}:
        db.execute(
            text("SELECT setval(pg_get_serial_sequence('participants', 'id'), 1, false)")
        )
        return True
    return False


def _next_public_participant_id(db: Session) -> str:
    max_participant_id = db.query(func.max(Participant.id)).scalar() or 0
    next_id = int(max_participant_id) + 1
    return f"R{next_id:04d}"


def reset_study_runtime(
    db: Session,
    *,
    campaign_id: int | None,
    all_active_campaigns: bool,
    all_campaigns: bool,
    remove_assets: bool,
    storage_root: Path,
) -> dict:
    campaign_ids = _resolve_target_campaign_ids(
        db,
        campaign_id=campaign_id,
        all_active_campaigns=all_active_campaigns,
        all_campaigns=all_campaigns,
    )
    deleted_counts, _ = _delete_runtime_rows(db, campaign_ids=campaign_ids)
    removed_asset_dirs = _remove_campaign_asset_dirs(storage_root, campaign_ids) if remove_assets else []
    identity_reset = _reset_participant_identity_if_empty(db)

    return {
        "campaign_ids": campaign_ids,
        "deleted_counts": {key: int(value) for key, value in deleted_counts.items()},
        "removed_asset_dirs": removed_asset_dirs,
        "next_participant_public_id": _next_public_participant_id(db),
        "identity_reset": identity_reset,
        "timestamp": _utcnow(),
    }
