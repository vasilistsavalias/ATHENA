from __future__ import annotations

import csv
import io
from collections import defaultdict
from itertools import combinations

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


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def _percent_agreement(values: list[str]) -> float | None:
    if len(values) < 2:
        return None
    total_pairs = 0
    agree_pairs = 0
    for a, b in combinations(values, 2):
        total_pairs += 1
        if a == b:
            agree_pairs += 1
    if total_pairs <= 0:
        return None
    return float(agree_pairs / total_pairs)


def _build_reliability_report(item_level_rows: list[dict], pairwise_rows: list[dict]) -> dict:
    block_a_by_item_auth: dict[str, list[int]] = defaultdict(list)
    block_a_by_item_plaus: dict[str, list[int]] = defaultdict(list)
    for row in item_level_rows:
        if row.get("block") != "A":
            continue
        item_key = str(row.get("item_id"))
        auth = row.get("authenticity_likelihood")
        plaus = row.get("archaeological_plausibility")
        if isinstance(auth, int):
            block_a_by_item_auth[item_key].append(auth)
        if isinstance(plaus, int):
            block_a_by_item_plaus[item_key].append(plaus)

    auth_pairwise = []
    plaus_pairwise = []
    for values in block_a_by_item_auth.values():
        if len(values) < 2:
            continue
        pairs = [1.0 if abs(a - b) <= 1 else 0.0 for a, b in combinations(values, 2)]
        if pairs:
            auth_pairwise.extend(pairs)
    for values in block_a_by_item_plaus.values():
        if len(values) < 2:
            continue
        pairs = [1.0 if abs(a - b) <= 1 else 0.0 for a, b in combinations(values, 2)]
        if pairs:
            plaus_pairwise.extend(pairs)

    block_b_by_item_choice: dict[str, list[str]] = defaultdict(list)
    for row in pairwise_rows:
        if row.get("is_practice"):
            continue
        choice = row.get("choice")
        if choice is None:
            continue
        item_key = str(row.get("item_id"))
        block_b_by_item_choice[item_key].append(str(choice))
    block_b_item_agreements = [
        agreement
        for choices in block_b_by_item_choice.values()
        for agreement in [_percent_agreement(choices)]
        if agreement is not None
    ]

    return {
        "block_a": {
            "items_with_multiple_raters_authenticity": sum(
                1 for v in block_a_by_item_auth.values() if len(v) >= 2
            ),
            "items_with_multiple_raters_plausibility": sum(
                1 for v in block_a_by_item_plaus.values() if len(v) >= 2
            ),
            "pairwise_within_onepoint_authenticity": _mean(auth_pairwise),
            "pairwise_within_onepoint_plausibility": _mean(plaus_pairwise),
        },
        "block_b": {
            "items_with_multiple_raters": sum(1 for v in block_b_by_item_choice.values() if len(v) >= 2),
            "mean_pairwise_percent_agreement": _mean(block_b_item_agreements),
        },
    }


def _build_exceedance_report(participant_rows: list[dict], pairwise_rows: list[dict]) -> dict:
    by_participant: dict[str, list[dict]] = defaultdict(list)
    for row in pairwise_rows:
        if row.get("is_practice") or row.get("is_anchor"):
            continue
        if row.get("restoration_preferred") is None:
            continue
        by_participant[str(row["participant_id"])].append(row)

    participant_signals = []
    for participant in participant_rows:
        public_id = str(participant["participant_id"])
        relevant_rows = by_participant.get(public_id, [])
        denominator = len(relevant_rows)
        restoration_preferences = sum(1 for row in relevant_rows if row.get("restoration_preferred") is True)
        rate = float(restoration_preferences / denominator) if denominator > 0 else None
        tier1 = bool(rate is not None and rate >= 0.4)
        participant_signals.append(
            {
                "participant_id": public_id,
                "comprehension_risk": bool(participant.get("comprehension_risk", False)),
                "non_anchor_scored_trials": denominator,
                "restoration_preferences": restoration_preferences,
                "restoration_preference_rate": rate,
                "tier_1_met": tier1,
            }
        )

    full_tier2_count = sum(1 for row in participant_signals if row["tier_1_met"])
    robust_subset = [row for row in participant_signals if not row["comprehension_risk"]]
    robust_tier2_count = sum(1 for row in robust_subset if row["tier_1_met"])

    return {
        "tier_1_threshold": 0.4,
        "tier_2_threshold": 3,
        "robustness_rule": "Tier 2 must hold both with and without comprehension-risk participants.",
        "exclusions": ["practice_items", "anchor_items", "tie_or_unsure_choices"],
        "full_cohort": {
            "participant_count": len(participant_signals),
            "participants_meeting_tier1": full_tier2_count,
            "tier_2_met": full_tier2_count >= 3,
        },
        "excluding_comprehension_risk": {
            "participant_count": len(robust_subset),
            "participants_meeting_tier1": robust_tier2_count,
            "tier_2_met": robust_tier2_count >= 3,
        },
        "findings_robust": full_tier2_count >= 3 and robust_tier2_count >= 3,
        "participant_signals": participant_signals,
    }


def _get_campaign(db: Session, campaign_id: int | None) -> Campaign:
    if campaign_id is not None:
        campaign = db.query(Campaign).filter(Campaign.id == campaign_id).first()
    else:
        campaign = db.query(Campaign).filter(Campaign.is_active.is_(True)).order_by(Campaign.id.desc()).first()
    if campaign is None:
        raise ValueError("Campaign not found.")
    return campaign


def build_export_bundle(db: Session, campaign_id: int | None = None) -> dict:
    campaign = _get_campaign(db, campaign_id)
    protocol_version = campaign.protocol_version or get_settings().expert_protocol_version

    participants = db.query(Participant).filter(Participant.campaign_id == campaign.id).order_by(Participant.id.asc()).all()
    feedback_rows = db.query(StageFeedback).filter(StageFeedback.campaign_id == campaign.id).all()
    feedback_by_participant = {
        (row.participant_id, row.block): row
        for row in feedback_rows
    }
    participant_rows = [
        {
            "campaign_id": campaign.id,
            "participant_id": p.public_id,
            "status": p.status,
            "protocol_version": protocol_version,
            "name": p.name,
            "institution": p.institution,
            "discipline": p.discipline,
            "discipline_other": p.discipline_other,
            "profile_completed_at": p.profile_completed_at.isoformat() if p.profile_completed_at else None,
            "block_b_comprehension_attempts": p.block_b_comprehension_attempts,
            "block_b_comprehension_passed": p.block_b_comprehension_passed_at is not None,
            "comprehension_risk": p.comprehension_risk,
            "block_a_stage_comment": (
                feedback_by_participant[(p.id, "A")].comment if (p.id, "A") in feedback_by_participant else None
            ),
            "block_b_stage_comment": (
                feedback_by_participant[(p.id, "B")].comment if (p.id, "B") in feedback_by_participant else None
            ),
            "block_c_stage_comment": (
                feedback_by_participant[(p.id, "C")].comment if (p.id, "C") in feedback_by_participant else None
            ),
            "created_at": p.created_at.isoformat(),
            "completed_at": p.completed_at.isoformat() if p.completed_at else None,
        }
        for p in participants
    ]

    item_level_rows = []
    pairwise_rows = []

    a_query = (
        db.query(BlockAResponse, BlockAAssignment, BlockAItem, Participant)
        .join(BlockAAssignment, BlockAResponse.assignment_id == BlockAAssignment.id)
        .join(BlockAItem, BlockAAssignment.item_id == BlockAItem.id)
        .join(Participant, BlockAResponse.participant_id == Participant.id)
        .filter(BlockAResponse.campaign_id == campaign.id)
    )
    for response, assignment, item, participant in a_query.all():
        metadata = item.metadata_json or {}
        item_level_rows.append(
            {
                "campaign_id": campaign.id,
                "participant_id": participant.public_id,
                "name": participant.name,
                "institution": participant.institution,
                "discipline": participant.discipline,
                "discipline_other": participant.discipline_other,
                "profile_completed_at": participant.profile_completed_at.isoformat() if participant.profile_completed_at else None,
                "block_a_stage_comment": (
                    feedback_by_participant[(participant.id, "A")].comment
                    if (participant.id, "A") in feedback_by_participant
                    else None
                ),
                "block_b_stage_comment": (
                    feedback_by_participant[(participant.id, "B")].comment
                    if (participant.id, "B") in feedback_by_participant
                    else None
                ),
                "block_c_stage_comment": (
                    feedback_by_participant[(participant.id, "C")].comment
                    if (participant.id, "C") in feedback_by_participant
                    else None
                ),
                "block": "A",
                "item_id": item.id,
                "item_order": assignment.item_order,
                "sample_id": item.sample_id,
                "mask_type": item.mask_type,
                "mask_coverage_bin": item.mask_coverage_bin,
                "is_practice": bool(metadata.get("is_practice", False)),
                "is_anchor": bool(metadata.get("is_anchor", False)),
                "is_attention_check": assignment.is_attention_check,
                "authenticity_likelihood": response.authenticity_likelihood,
                "archaeological_plausibility": response.archaeological_plausibility,
                "choice": None,
                "confidence": response.confidence,
                "response_time_ms": response.response_time_ms,
                "comment": response.comment,
                "created_at": response.created_at.isoformat(),
            }
        )

    b_query = (
        db.query(BlockBResponse, BlockBAssignment, BlockBItem, Participant)
        .join(BlockBAssignment, BlockBResponse.assignment_id == BlockBAssignment.id)
        .join(BlockBItem, BlockBAssignment.item_id == BlockBItem.id)
        .join(Participant, BlockBResponse.participant_id == Participant.id)
        .filter(BlockBResponse.campaign_id == campaign.id)
    )
    for response, assignment, item, participant in b_query.all():
        metadata = item.metadata_json or {}
        block_part = str(metadata.get("block_part") or "B").upper()
        block_label = "C" if block_part == "C" else "B"
        row = {
            "campaign_id": campaign.id,
            "participant_id": participant.public_id,
            "name": participant.name,
            "institution": participant.institution,
            "discipline": participant.discipline,
            "discipline_other": participant.discipline_other,
            "profile_completed_at": participant.profile_completed_at.isoformat() if participant.profile_completed_at else None,
            "block_a_stage_comment": (
                feedback_by_participant[(participant.id, "A")].comment
                if (participant.id, "A") in feedback_by_participant
                else None
            ),
            "block_b_stage_comment": (
                feedback_by_participant[(participant.id, "B")].comment
                if (participant.id, "B") in feedback_by_participant
                else None
            ),
            "block_c_stage_comment": (
                feedback_by_participant[(participant.id, "C")].comment
                if (participant.id, "C") in feedback_by_participant
                else None
            ),
            "block": block_label,
            "item_id": item.id,
            "item_order": assignment.item_order,
            "sample_id": item.sample_id,
            "mask_type": item.mask_type,
            "mask_coverage_bin": item.mask_coverage_bin,
            "is_practice": bool(metadata.get("is_practice", False)),
            "is_anchor": bool(metadata.get("is_anchor", False)),
            "is_attention_check": assignment.is_attention_check,
            "block_part": block_label,
            "calibration_choice": metadata.get("calibration_choice"),
            "restoration_choice": metadata.get("restoration_choice"),
            "restoration_preferred": None,
            "authenticity_likelihood": None,
            "archaeological_plausibility": None,
            "choice": response.choice,
            "confidence": response.confidence,
            "response_time_ms": response.response_time_ms,
            "comment": response.comment,
            "created_at": response.created_at.isoformat(),
        }
        item_level_rows.append(row)
        pairwise_rows.append(row)

    for row in pairwise_rows:
        choice = row.get("choice")
        restoration_choice = row.get("restoration_choice")
        if choice in {"A", "B"} and restoration_choice in {"A", "B"}:
            row["restoration_preferred"] = choice == restoration_choice
        else:
            row["restoration_preferred"] = None

    flags = (
        db.query(AttentionFlag, Participant)
        .join(Participant, AttentionFlag.participant_id == Participant.id)
        .filter(AttentionFlag.campaign_id == campaign.id)
        .all()
    )
    quality_flags_rows = [
        {
            "campaign_id": campaign.id,
            "participant_id": participant.public_id,
            "block": flag.block,
            "assignment_id": flag.assignment_id,
            "flag_type": flag.flag_type,
            "details": flag.details_json,
            "created_at": flag.created_at.isoformat(),
        }
        for flag, participant in flags
    ]

    by_participant = defaultdict(int)
    for row in quality_flags_rows:
        by_participant[row["participant_id"]] += 1

    quality_report = {
        "campaign_id": campaign.id,
        "campaign_name": campaign.name,
        "participants_total": len(participant_rows),
        "responses_total": len(item_level_rows),
        "profiles_completed_total": sum(1 for row in participant_rows if row["profile_completed_at"]),
        "block_a_feedback_total": sum(1 for row in participant_rows if row["block_a_stage_comment"]),
        "block_b_feedback_total": sum(1 for row in participant_rows if row["block_b_stage_comment"]),
        "block_c_feedback_total": sum(1 for row in participant_rows if row["block_c_stage_comment"]),
        "attention_flags_total": len(quality_flags_rows),
        "comprehension_risk_total": sum(1 for row in participant_rows if row["comprehension_risk"]),
        "reliability": _build_reliability_report(item_level_rows, pairwise_rows),
        "expert_plausibility_exceedance": _build_exceedance_report(participant_rows, pairwise_rows),
        "flagged_participants": [
            {"participant_id": pid, "flags": count}
            for pid, count in sorted(by_participant.items(), key=lambda x: (-x[1], x[0]))
        ],
    }

    return {
        "campaign": {
            "id": campaign.id,
            "name": campaign.name,
            "seed": campaign.seed,
            "protocol_version": protocol_version,
            "block_a_target_count": campaign.block_a_target_count,
            "block_b_target_count": campaign.block_b_target_count,
            "block_c_target_count": sum(
                1
                for row in db.query(BlockBItem).filter(BlockBItem.campaign_id == campaign.id).all()
                if str((row.metadata_json or {}).get("block_part") or "B").upper() == "C"
            ),
        },
        "participant_level": participant_rows,
        "stage_feedback": [
            {
                "campaign_id": campaign.id,
                "participant_id": next(
                    (participant.public_id for participant in participants if participant.id == row.participant_id),
                    None,
                ),
                "block": row.block,
                "comment": row.comment,
                "created_at": row.created_at.isoformat(),
                "updated_at": row.updated_at.isoformat(),
            }
            for row in feedback_rows
        ],
        "item_level": item_level_rows,
        "pairwise_preference": pairwise_rows,
        "quality_flags": quality_flags_rows,
        "quality_report": quality_report,
    }


def bundle_to_csv(bundle: dict) -> str:
    rows = bundle.get("item_level", [])
    if not rows:
        return ""
    columns = [
        "campaign_id",
        "participant_id",
        "name",
        "institution",
        "discipline",
        "discipline_other",
        "profile_completed_at",
        "block_a_stage_comment",
        "block_b_stage_comment",
        "block_c_stage_comment",
        "block",
        "item_id",
        "item_order",
        "sample_id",
        "mask_type",
        "mask_coverage_bin",
        "is_practice",
        "is_anchor",
        "is_attention_check",
        "block_part",
        "calibration_choice",
        "restoration_choice",
        "restoration_preferred",
        "authenticity_likelihood",
        "archaeological_plausibility",
        "choice",
        "confidence",
        "response_time_ms",
        "comment",
        "created_at",
    ]
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=columns)
    writer.writeheader()
    for row in rows:
        writer.writerow({key: row.get(key) for key in columns})
    return buffer.getvalue()
