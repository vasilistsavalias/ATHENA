from __future__ import annotations

import random
import hashlib
from datetime import datetime, timezone

from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.models import (
    AttentionFlag,
    BlockAAssignment,
    BlockAItem,
    BlockAResponse,
    BlockBAssignment,
    BlockBItem,
    BlockBResponse,
    Campaign,
    Participant,
    StageFeedback,
)


def _utcnow():
    return datetime.now(timezone.utc)


def _block_part_from_metadata(metadata_json: dict | None) -> str:
    part = str((metadata_json or {}).get("block_part") or "B").upper()
    return "C" if part == "C" else "B"


def _rng(*, seed: int, participant_id: int, salt: str) -> random.Random:
    token = f"{int(seed)}:{int(participant_id)}:{salt}".encode("utf-8")
    digest = hashlib.sha256(token).hexdigest()[:16]
    mixed = int(digest, 16) & 0xFFFFFFFF
    return random.Random(mixed)


def ensure_assignments(db: Session, campaign: Campaign, participant: Participant):
    settings = get_settings()
    target_a = campaign.block_a_target_count if campaign.block_a_target_count is not None else settings.block_a_target_count
    target_b = campaign.block_b_target_count if campaign.block_b_target_count is not None else settings.block_b_target_count
    a_count = db.query(BlockAAssignment).filter_by(campaign_id=campaign.id, participant_id=participant.id).count()
    b_count = db.query(BlockBAssignment).filter_by(campaign_id=campaign.id, participant_id=participant.id).count()
    a_ready = target_a <= 0 or a_count > 0
    b_ready = target_b <= 0 or b_count > 0
    if a_ready and b_ready:
        return

    if a_count == 0 and target_a > 0:
        _create_block_a_assignments(
            db,
            campaign=campaign,
            participant=participant,
            target_count=target_a,
            attention_count=settings.block_a_attention_checks,
        )

    if b_count == 0 and target_b > 0:
        _create_block_b_assignments(
            db,
            campaign=campaign,
            participant=participant,
            target_count=target_b,
            attention_count=settings.block_b_attention_checks,
        )

    db.commit()


def _create_block_a_assignments(
    db: Session,
    *,
    campaign: Campaign,
    participant: Participant,
    target_count: int,
    attention_count: int,
):
    if int(target_count) <= 0:
        return
    target_count = max(1, int(target_count))
    attention_count = max(0, min(int(attention_count), target_count - 1))
    base_count = target_count - attention_count

    items = list(db.query(BlockAItem).filter_by(campaign_id=campaign.id).order_by(BlockAItem.id.asc()).all())
    if len(items) < base_count:
        raise ValueError(f"Not enough Block A items ({len(items)}) for target count {target_count}.")

    rng = _rng(seed=campaign.seed, participant_id=participant.id, salt="block-a")
    rng.shuffle(items)
    selected = items[:base_count]
    assignments: list[BlockAAssignment] = []

    order = 1
    for item in selected:
        assignments.append(
            BlockAAssignment(
                campaign_id=campaign.id,
                participant_id=participant.id,
                item_id=item.id,
                item_order=order,
                is_attention_check=False,
            )
        )
        order += 1

    if attention_count > 0:
        attention_sources = selected[:attention_count]
        for source in attention_sources:
            assignments.append(
                BlockAAssignment(
                    campaign_id=campaign.id,
                    participant_id=participant.id,
                    item_id=source.id,
                    item_order=order,
                    is_attention_check=True,
                    attention_source_item_id=source.id,
                )
            )
            order += 1

    db.add_all(assignments)


def _create_block_b_assignments(
    db: Session,
    *,
    campaign: Campaign,
    participant: Participant,
    target_count: int,
    attention_count: int,
):
    if int(target_count) <= 0:
        return
    target_count = max(1, int(target_count))

    items = list(db.query(BlockBItem).filter_by(campaign_id=campaign.id).order_by(BlockBItem.id.asc()).all())
    if not items:
        raise ValueError(f"Not enough Block B/C items (0) for target count {target_count}.")
    b_items = [item for item in items if _block_part_from_metadata(item.metadata_json) == "B"]
    c_items = [item for item in items if _block_part_from_metadata(item.metadata_json) == "C"]
    study_mode = None
    if items and isinstance(items[0].metadata_json, dict):
        study_mode = items[0].metadata_json.get("study_mode")

    # Three-part campaigns are encoded with Block B + Block C items sharing the same table.
    # Keep part ordering deterministic: all B first, then all C.
    if c_items:
        rng_b = _rng(seed=campaign.seed, participant_id=participant.id, salt="block-b")
        rng_c = _rng(seed=campaign.seed, participant_id=participant.id, salt="block-c")
        b_practice = [item for item in b_items if bool((item.metadata_json or {}).get("is_practice", False))]
        b_scored = [item for item in b_items if not bool((item.metadata_json or {}).get("is_practice", False))]
        rng_b.shuffle(b_scored)
        rng_c.shuffle(c_items)
        ordered_items = b_practice + b_scored + c_items
        if len(ordered_items) < target_count:
            raise ValueError(f"Not enough Block B/C items ({len(ordered_items)}) for target count {target_count}.")
        assignments: list[BlockBAssignment] = []
        for order, item in enumerate(ordered_items[:target_count], start=1):
            show_a_left = (rng_c if _block_part_from_metadata(item.metadata_json) == "C" else rng_b).random() < 0.5
            assignments.append(
                BlockBAssignment(
                    campaign_id=campaign.id,
                    participant_id=participant.id,
                    item_id=item.id,
                    item_order=order,
                    show_a_left=show_a_left,
                    is_attention_check=False,
                )
            )
        db.add_all(assignments)
        return

    if study_mode == "pairwise_only":
        practice_items = [item for item in b_items if bool((item.metadata_json or {}).get("is_practice", False))]
        scored_items = [item for item in b_items if not bool((item.metadata_json or {}).get("is_practice", False))]
        if len(b_items) < target_count:
            raise ValueError(f"Not enough Block B items ({len(b_items)}) for target count {target_count}.")

        rng = _rng(seed=campaign.seed, participant_id=participant.id, salt="block-b")
        rng.shuffle(scored_items)
        selected_scored = scored_items[: max(0, target_count - len(practice_items))]
        ordered_items = practice_items + selected_scored
        if len(ordered_items) < target_count:
            remaining = [item for item in scored_items if item.id not in {x.id for x in selected_scored}]
            ordered_items.extend(remaining[: target_count - len(ordered_items)])

        assignments: list[BlockBAssignment] = []
        for order, item in enumerate(ordered_items[:target_count], start=1):
            show_a_left = rng.random() < 0.5
            assignments.append(
                BlockBAssignment(
                    campaign_id=campaign.id,
                    participant_id=participant.id,
                    item_id=item.id,
                    item_order=order,
                    show_a_left=show_a_left,
                    is_attention_check=False,
                )
            )
        db.add_all(assignments)
        return

    attention_count = max(0, min(int(attention_count), target_count - 1))
    base_count = target_count - attention_count

    if len(b_items) < base_count:
        raise ValueError(f"Not enough Block B items ({len(b_items)}) for target count {target_count}.")

    rng = _rng(seed=campaign.seed, participant_id=participant.id, salt="block-b")
    rng.shuffle(b_items)
    selected = b_items[:base_count]
    assignments: list[BlockBAssignment] = []

    order = 1
    for item in selected:
        show_a_left = rng.random() < 0.5
        assignments.append(
            BlockBAssignment(
                campaign_id=campaign.id,
                participant_id=participant.id,
                item_id=item.id,
                item_order=order,
                show_a_left=show_a_left,
                is_attention_check=False,
            )
        )
        order += 1

    if attention_count > 0:
        attention_sources = selected[:attention_count]
        for source in attention_sources:
            show_a_left = rng.random() < 0.5
            assignments.append(
                BlockBAssignment(
                    campaign_id=campaign.id,
                    participant_id=participant.id,
                    item_id=source.id,
                    item_order=order,
                    show_a_left=show_a_left,
                    is_attention_check=True,
                    attention_source_item_id=source.id,
                )
            )
            order += 1

    db.add_all(assignments)


def get_next_block_a_assignment(db: Session, *, campaign_id: int, participant_id: int) -> BlockAAssignment | None:
    return (
        db.query(BlockAAssignment)
        .filter(
            BlockAAssignment.campaign_id == campaign_id,
            BlockAAssignment.participant_id == participant_id,
            BlockAAssignment.completed_at.is_(None),
        )
        .order_by(BlockAAssignment.item_order.asc())
        .first()
    )


def get_next_block_b_assignment(db: Session, *, campaign_id: int, participant_id: int) -> BlockBAssignment | None:
    return get_next_block_assignment_by_part(db, campaign_id=campaign_id, participant_id=participant_id, part="B")


def get_next_block_assignment_by_part(
    db: Session,
    *,
    campaign_id: int,
    participant_id: int,
    part: str,
) -> BlockBAssignment | None:
    desired = "C" if str(part).upper() == "C" else "B"
    candidates = (
        db.query(BlockBAssignment)
        .filter(
            BlockBAssignment.campaign_id == campaign_id,
            BlockBAssignment.participant_id == participant_id,
            BlockBAssignment.completed_at.is_(None),
        )
        .order_by(BlockBAssignment.item_order.asc())
        .all()
    )
    for assignment in candidates:
        if _block_part_from_metadata(assignment.item.metadata_json) == desired:
            return assignment
    return None


def submit_block_a_response(
    db: Session,
    *,
    assignment: BlockAAssignment,
    participant: Participant,
    authenticity_likelihood: int,
    archaeological_plausibility: int,
    confidence: int,
    comment: str | None,
    response_time_ms: int,
):
    existing = db.query(BlockAResponse).filter_by(assignment_id=assignment.id).first()
    if existing is not None:
        raise ValueError("Assignment already submitted.")
    response = BlockAResponse(
        campaign_id=participant.campaign_id,
        participant_id=participant.id,
        assignment_id=assignment.id,
        authenticity_likelihood=authenticity_likelihood,
        archaeological_plausibility=archaeological_plausibility,
        confidence=confidence,
        comment=comment,
        response_time_ms=response_time_ms,
    )
    assignment.completed_at = _utcnow()
    db.add(response)
    _run_block_a_attention_check(db, assignment=assignment, response=response, participant=participant)
    db.commit()


def submit_block_b_response(
    db: Session,
    *,
    assignment: BlockBAssignment,
    participant: Participant,
    choice: str,
    confidence: int,
    comment: str | None,
    response_time_ms: int,
):
    existing = db.query(BlockBResponse).filter_by(assignment_id=assignment.id).first()
    if existing is not None:
        raise ValueError("Assignment already submitted.")
    response = BlockBResponse(
        campaign_id=participant.campaign_id,
        participant_id=participant.id,
        assignment_id=assignment.id,
        choice=choice,
        confidence=confidence,
        comment=comment,
        response_time_ms=response_time_ms,
    )
    assignment.completed_at = _utcnow()
    db.add(response)
    _run_block_b_attention_check(db, assignment=assignment, response=response, participant=participant)
    db.commit()


def _run_block_a_attention_check(
    db: Session,
    *,
    assignment: BlockAAssignment,
    response: BlockAResponse,
    participant: Participant,
):
    if not assignment.is_attention_check or assignment.attention_source_item_id is None:
        return
    source_assignment = (
        db.query(BlockAAssignment)
        .filter(
            BlockAAssignment.campaign_id == assignment.campaign_id,
            BlockAAssignment.participant_id == assignment.participant_id,
            BlockAAssignment.item_id == assignment.attention_source_item_id,
            BlockAAssignment.is_attention_check.is_(False),
        )
        .order_by(BlockAAssignment.item_order.asc())
        .first()
    )
    if source_assignment is None:
        return
    source_response = db.query(BlockAResponse).filter_by(assignment_id=source_assignment.id).first()
    if source_response is None:
        return
    auth_diff = abs(source_response.authenticity_likelihood - response.authenticity_likelihood)
    plaus_diff = abs(source_response.archaeological_plausibility - response.archaeological_plausibility)
    if auth_diff >= 3 or plaus_diff >= 3:
        db.add(
            AttentionFlag(
                campaign_id=participant.campaign_id,
                participant_id=participant.id,
                block="A",
                assignment_id=assignment.id,
                flag_type="inconsistent_repeat",
                details_json={
                    "source_assignment_id": source_assignment.id,
                    "auth_diff": auth_diff,
                    "plausibility_diff": plaus_diff,
                },
            )
        )


def _run_block_b_attention_check(
    db: Session,
    *,
    assignment: BlockBAssignment,
    response: BlockBResponse,
    participant: Participant,
):
    if not assignment.is_attention_check or assignment.attention_source_item_id is None:
        return
    source_assignment = (
        db.query(BlockBAssignment)
        .filter(
            BlockBAssignment.campaign_id == assignment.campaign_id,
            BlockBAssignment.participant_id == assignment.participant_id,
            BlockBAssignment.item_id == assignment.attention_source_item_id,
            BlockBAssignment.is_attention_check.is_(False),
        )
        .order_by(BlockBAssignment.item_order.asc())
        .first()
    )
    if source_assignment is None:
        return
    source_response = db.query(BlockBResponse).filter_by(assignment_id=source_assignment.id).first()
    if source_response is None:
        return
    if source_response.choice != response.choice:
        db.add(
            AttentionFlag(
                campaign_id=participant.campaign_id,
                participant_id=participant.id,
                block="B",
                assignment_id=assignment.id,
                flag_type="inconsistent_repeat",
                details_json={
                    "source_assignment_id": source_assignment.id,
                    "source_choice": source_response.choice,
                    "repeat_choice": response.choice,
                },
            )
        )


def compute_progress(db: Session, *, campaign_id: int, participant_id: int):
    block_a_total = db.query(BlockAAssignment).filter_by(campaign_id=campaign_id, participant_id=participant_id).count()
    b_assignments = (
        db.query(BlockBAssignment)
        .filter_by(campaign_id=campaign_id, participant_id=participant_id)
        .all()
    )
    b_responses = (
        db.query(BlockBResponse)
        .filter_by(campaign_id=campaign_id, participant_id=participant_id)
        .all()
    )
    responded_ids = {row.assignment_id for row in b_responses}
    block_b_total = sum(1 for row in b_assignments if _block_part_from_metadata(row.item.metadata_json) == "B")
    block_c_total = sum(1 for row in b_assignments if _block_part_from_metadata(row.item.metadata_json) == "C")
    block_a_completed = (
        db.query(BlockAResponse)
        .filter_by(campaign_id=campaign_id, participant_id=participant_id)
        .count()
    )
    block_b_completed = sum(
        1
        for row in b_assignments
        if _block_part_from_metadata(row.item.metadata_json) == "B" and row.id in responded_ids
    )
    block_c_completed = sum(
        1
        for row in b_assignments
        if _block_part_from_metadata(row.item.metadata_json) == "C" and row.id in responded_ids
    )
    participant = (
        db.query(Participant)
        .filter(Participant.campaign_id == campaign_id, Participant.id == participant_id)
        .first()
    )
    feedback_blocks = {
        row.block
        for row in db.query(StageFeedback)
        .filter(StageFeedback.campaign_id == campaign_id, StageFeedback.participant_id == participant_id)
        .all()
    }
    profile_completed = participant.profile_completed_at is not None if participant is not None else False
    block_a_feedback_completed = block_a_total == 0 or "A" in feedback_blocks
    block_b_feedback_completed = block_b_total == 0 or "B" in feedback_blocks
    block_c_feedback_completed = block_c_total == 0 or "C" in feedback_blocks
    return {
        "block_a_completed": block_a_completed,
        "block_a_total": block_a_total,
        "block_b_completed": block_b_completed,
        "block_b_total": block_b_total,
        "block_c_completed": block_c_completed,
        "block_c_total": block_c_total,
        "profile_completed": profile_completed,
        "block_a_feedback_completed": block_a_feedback_completed,
        "block_b_feedback_completed": block_b_feedback_completed,
        "block_c_feedback_completed": block_c_feedback_completed,
        "is_complete": (
            block_a_completed >= block_a_total
            and block_b_completed >= block_b_total
            and block_c_completed >= block_c_total
            and block_a_feedback_completed
            and block_b_feedback_completed
            and block_c_feedback_completed
        ),
    }
