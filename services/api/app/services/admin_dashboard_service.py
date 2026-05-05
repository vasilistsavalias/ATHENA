from __future__ import annotations

from collections import Counter

from sqlalchemy import func
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


def _get_campaign(db: Session, campaign_id: int | None) -> Campaign:
    if campaign_id is not None:
        campaign = db.query(Campaign).filter(Campaign.id == campaign_id).first()
    else:
        campaign = db.query(Campaign).filter(Campaign.is_active.is_(True)).order_by(Campaign.id.desc()).first()
    if campaign is None:
        raise ValueError("Campaign not found.")
    return campaign


def build_admin_dashboard(db: Session, campaign_id: int | None = None) -> dict:
    try:
        campaign = _get_campaign(db, campaign_id)
    except ValueError:
        if campaign_id is not None:
            raise
        return {
            "campaign": None,
            "stats": {
                "participants_total": 0,
                "participants_completed": 0,
                "participants_active": 0,
                "profiles_completed": 0,
                "block_a_feedback_completed": 0,
                "block_b_feedback_completed": 0,
                "block_c_feedback_completed": 0,
                "block_a_responses": 0,
                "block_b_responses": 0,
                "block_c_responses": 0,
                "attention_flags_total": 0,
                "comprehension_risk_total": 0,
            },
            "discipline_breakdown": [],
            "participants": [],
            "recent_stage_feedback": [],
            "recent_item_comments": [],
        }
    protocol_version = campaign.protocol_version or get_settings().expert_protocol_version
    participants = (
        db.query(Participant)
        .filter(Participant.campaign_id == campaign.id)
        .order_by(Participant.created_at.desc(), Participant.id.desc())
        .all()
    )
    participant_ids = [participant.id for participant in participants]

    block_a_counts = dict(
        db.query(BlockAResponse.participant_id, func.count(BlockAResponse.id))
        .filter(BlockAResponse.campaign_id == campaign.id)
        .group_by(BlockAResponse.participant_id)
        .all()
    )
    block_b_assignment_rows = (
        db.query(BlockBAssignment, BlockBItem)
        .join(BlockBItem, BlockBAssignment.item_id == BlockBItem.id)
        .filter(BlockBAssignment.campaign_id == campaign.id)
        .all()
    )
    block_b_totals: dict[int, int] = {}
    block_c_totals: dict[int, int] = {}
    for assignment, item in block_b_assignment_rows:
        part = str((item.metadata_json or {}).get("block_part") or "B").upper()
        if part == "C":
            block_c_totals[assignment.participant_id] = int(block_c_totals.get(assignment.participant_id, 0)) + 1
        else:
            block_b_totals[assignment.participant_id] = int(block_b_totals.get(assignment.participant_id, 0)) + 1

    block_b_counts: dict[int, int] = {}
    block_c_counts: dict[int, int] = {}
    block_b_response_rows = (
        db.query(BlockBResponse, BlockBAssignment, BlockBItem)
        .join(BlockBAssignment, BlockBResponse.assignment_id == BlockBAssignment.id)
        .join(BlockBItem, BlockBAssignment.item_id == BlockBItem.id)
        .filter(BlockBResponse.campaign_id == campaign.id)
        .all()
    )
    for _, assignment, item in block_b_response_rows:
        part = str((item.metadata_json or {}).get("block_part") or "B").upper()
        if part == "C":
            block_c_counts[assignment.participant_id] = int(block_c_counts.get(assignment.participant_id, 0)) + 1
        else:
            block_b_counts[assignment.participant_id] = int(block_b_counts.get(assignment.participant_id, 0)) + 1
    flag_counts = dict(
        db.query(AttentionFlag.participant_id, func.count(AttentionFlag.id))
        .filter(AttentionFlag.campaign_id == campaign.id)
        .group_by(AttentionFlag.participant_id)
        .all()
    )
    feedback_rows = db.query(StageFeedback).filter(StageFeedback.campaign_id == campaign.id).all()
    feedback_map = {(row.participant_id, row.block): row for row in feedback_rows}

    discipline_counter: Counter[str] = Counter()
    participant_rows: list[dict] = []
    for participant in participants:
        discipline_label = participant.discipline or "Not provided"
        discipline_counter[discipline_label] += 1
        participant_rows.append(
            {
                "participant_id": participant.public_id,
                "status": participant.status,
                "name": participant.name,
                "institution": participant.institution,
                "discipline": participant.discipline,
                "profile_completed": participant.profile_completed_at is not None,
                "block_b_comprehension_attempts": participant.block_b_comprehension_attempts,
                "block_b_comprehension_passed": participant.block_b_comprehension_passed_at is not None,
                "comprehension_risk": participant.comprehension_risk,
                "block_a_completed": int(block_a_counts.get(participant.id, 0)),
                "block_a_total": campaign.block_a_target_count,
                "block_b_completed": int(block_b_counts.get(participant.id, 0)),
                "block_b_total": int(block_b_totals.get(participant.id, 0)),
                "block_c_completed": int(block_c_counts.get(participant.id, 0)),
                "block_c_total": int(block_c_totals.get(participant.id, 0)),
                "block_a_feedback_completed": (participant.id, "A") in feedback_map,
                "block_b_feedback_completed": (participant.id, "B") in feedback_map,
                "block_c_feedback_completed": (participant.id, "C") in feedback_map,
                "attention_flags": int(flag_counts.get(participant.id, 0)),
                "block_a_stage_comment": feedback_map[(participant.id, "A")].comment if (participant.id, "A") in feedback_map else None,
                "block_b_stage_comment": feedback_map[(participant.id, "B")].comment if (participant.id, "B") in feedback_map else None,
                "block_c_stage_comment": feedback_map[(participant.id, "C")].comment if (participant.id, "C") in feedback_map else None,
                "created_at": participant.created_at,
                "completed_at": participant.completed_at,
            }
        )

    recent_stage_feedback = [
        {
            "participant_id": participant.public_id,
            "block": row.block,
            "source": "stage_feedback",
            "comment": row.comment,
            "sample_id": None,
            "created_at": row.updated_at,
        }
        for row, participant in (
            db.query(StageFeedback, Participant)
            .join(Participant, StageFeedback.participant_id == Participant.id)
            .filter(StageFeedback.campaign_id == campaign.id)
            .order_by(StageFeedback.updated_at.desc())
            .limit(10)
            .all()
        )
    ]

    block_a_comments = [
        {
            "participant_id": participant.public_id,
            "block": "A",
            "source": "item_comment",
            "comment": response.comment,
            "sample_id": item.sample_id,
            "created_at": response.created_at,
        }
        for response, participant, item in (
            db.query(BlockAResponse, Participant, BlockAItem)
            .join(Participant, BlockAResponse.participant_id == Participant.id)
            .join(BlockAAssignment, BlockAResponse.assignment_id == BlockAAssignment.id)
            .join(BlockAItem, BlockAAssignment.item_id == BlockAItem.id)
            .filter(
                BlockAResponse.campaign_id == campaign.id,
                BlockAResponse.comment.isnot(None),
                BlockAResponse.comment != "",
            )
            .order_by(BlockAResponse.created_at.desc())
            .limit(10)
            .all()
        )
    ]
    block_b_comments = [
        {
            "participant_id": participant.public_id,
            "block": "C" if str((item.metadata_json or {}).get("block_part") or "B").upper() == "C" else "B",
            "source": "item_comment",
            "comment": response.comment,
            "sample_id": item.sample_id,
            "created_at": response.created_at,
        }
        for response, participant, item in (
            db.query(BlockBResponse, Participant, BlockBItem)
            .join(Participant, BlockBResponse.participant_id == Participant.id)
            .join(BlockBAssignment, BlockBResponse.assignment_id == BlockBAssignment.id)
            .join(BlockBItem, BlockBAssignment.item_id == BlockBItem.id)
            .filter(
                BlockBResponse.campaign_id == campaign.id,
                BlockBResponse.comment.isnot(None),
                BlockBResponse.comment != "",
            )
            .order_by(BlockBResponse.created_at.desc())
            .limit(10)
            .all()
        )
    ]
    recent_item_comments = sorted(block_a_comments + block_b_comments, key=lambda row: row["created_at"], reverse=True)[:12]

    participants_completed = sum(1 for row in participant_rows if row["status"] == "completed")
    stats = {
        "participants_total": len(participant_rows),
        "participants_completed": participants_completed,
        "participants_active": len(participant_rows) - participants_completed,
        "profiles_completed": sum(1 for row in participant_rows if row["profile_completed"]),
        "block_a_feedback_completed": sum(1 for row in participant_rows if row["block_a_feedback_completed"]),
        "block_b_feedback_completed": sum(1 for row in participant_rows if row["block_b_feedback_completed"]),
        "block_c_feedback_completed": sum(1 for row in participant_rows if row["block_c_feedback_completed"]),
        "block_a_responses": sum(int(value) for value in block_a_counts.values()),
        "block_b_responses": sum(int(value) for value in block_b_counts.values()),
        "block_c_responses": sum(int(value) for value in block_c_counts.values()),
        "attention_flags_total": sum(int(value) for value in flag_counts.values()),
        "comprehension_risk_total": sum(1 for row in participant_rows if row["comprehension_risk"]),
    }

    return {
        "campaign": {
            "id": campaign.id,
            "name": campaign.name,
            "seed": campaign.seed,
            "protocol_version": protocol_version,
            "block_a_target_count": campaign.block_a_target_count,
            "block_b_target_count": campaign.block_b_target_count,
        },
        "stats": stats,
        "discipline_breakdown": [
            {"discipline": discipline, "count": count}
            for discipline, count in sorted(discipline_counter.items(), key=lambda entry: (-entry[1], entry[0]))
        ],
        "participants": participant_rows,
        "recent_stage_feedback": recent_stage_feedback,
        "recent_item_comments": recent_item_comments,
    }
