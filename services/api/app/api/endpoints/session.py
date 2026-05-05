from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.api.deps import get_current_participant
from app.core.config import get_settings
from app.db.session import get_db
from app.models import BlockBItem, Campaign, Participant
from app.schemas.session import (
    BlockBComprehensionRequest,
    BlockBComprehensionResponse,
    CampaignInfo,
    ProgressResponse,
    SessionInfo,
    SessionProfileResponse,
    SessionProfileUpdateRequest,
    StageFeedbackResponse,
    StageFeedbackUpdateRequest,
)
from app.services.assignment_service import compute_progress, ensure_assignments
from app.services.audit_service import log_event
from app.services.feedback_service import get_stage_feedback, upsert_stage_feedback

router = APIRouter()

DISCIPLINE_VALUES = {
    "Archaeology",
    "Philology / History / Archaeology",
    "Conservation / Restoration",
    "Museum / Curatorial",
    "Other",
}

BLOCK_B_COMPREHENSION_CORRECT_OPTION = "spot_machine"
BLOCK_B_COMPREHENSION_MAX_ATTEMPTS = 2


def _campaign_or_404(db: Session, campaign_id: int) -> Campaign:
    campaign = db.query(Campaign).filter(Campaign.id == campaign_id).first()
    if campaign is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Campaign not found.")
    return campaign


def _block_c_target_count(db: Session, campaign_id: int) -> int:
    total = 0
    for row in db.query(BlockBItem).filter(BlockBItem.campaign_id == campaign_id).all():
        part = str((row.metadata_json or {}).get("block_part") or "B").upper()
        if part == "C":
            total += 1
    return total


@router.get("/me", response_model=SessionInfo)
def session_me(
    db: Session = Depends(get_db),
    participant: Participant = Depends(get_current_participant),
):
    campaign = _campaign_or_404(db, participant.campaign_id)
    settings = get_settings()
    return SessionInfo(
        participant_id=participant.public_id,
        status=participant.status,
        profile_completed=participant.profile_completed_at is not None,
        created_at=participant.created_at,
        completed_at=participant.completed_at,
        campaign=CampaignInfo(
            id=campaign.id,
            name=campaign.name,
            seed=campaign.seed,
            protocol_version=campaign.protocol_version or settings.expert_protocol_version,
            block_a_target_count=campaign.block_a_target_count,
            block_b_target_count=campaign.block_b_target_count,
            block_c_target_count=_block_c_target_count(db, campaign.id),
        ),
        block_b_comprehension_attempts=participant.block_b_comprehension_attempts,
        block_b_comprehension_passed=participant.block_b_comprehension_passed_at is not None,
        comprehension_risk=participant.comprehension_risk,
    )


@router.get("/progress", response_model=ProgressResponse)
def progress(
    db: Session = Depends(get_db),
    participant: Participant = Depends(get_current_participant),
):
    ensure_assignments(db, campaign=participant.campaign, participant=participant)
    progress_data = compute_progress(db, campaign_id=participant.campaign_id, participant_id=participant.id)
    return ProgressResponse(**progress_data)


@router.post("/complete", response_model=ProgressResponse)
def complete_session(
    db: Session = Depends(get_db),
    participant: Participant = Depends(get_current_participant),
):
    progress_data = compute_progress(db, campaign_id=participant.campaign_id, participant_id=participant.id)
    if not progress_data["is_complete"]:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Session is not complete yet.")

    participant.status = "completed"
    participant.completed_at = datetime.now(timezone.utc)
    log_event(
        db,
        action="session.complete",
        campaign_id=participant.campaign_id,
        participant_id=participant.id,
        payload={"participant_public_id": participant.public_id},
    )
    db.commit()
    return ProgressResponse(**progress_data)


@router.get("/profile", response_model=SessionProfileResponse)
def session_profile(
    participant: Participant = Depends(get_current_participant),
):
    return SessionProfileResponse(
        name=participant.name,
        institution=participant.institution,
        discipline=participant.discipline,
        discipline_other=participant.discipline_other,
        profile_completed=participant.profile_completed_at is not None,
    )


@router.put("/profile", response_model=SessionProfileResponse)
def update_session_profile(
    payload: SessionProfileUpdateRequest,
    db: Session = Depends(get_db),
    participant: Participant = Depends(get_current_participant),
):
    discipline = (payload.discipline or "Archaeology").strip()
    if discipline not in DISCIPLINE_VALUES:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Invalid discipline.")

    name = (payload.name or "").strip()
    institution = (payload.institution or "").strip()
    if not name:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Name is required.")
    if not institution:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Institution is required.")

    discipline_other = (payload.discipline_other or "").strip() or None
    if discipline != "Other":
        discipline_other = None

    participant.name = name
    participant.institution = institution
    participant.discipline = discipline
    participant.discipline_other = discipline_other
    participant.profile_completed_at = participant.profile_completed_at or datetime.now(timezone.utc)
    log_event(
        db,
        action="session.profile.update",
        campaign_id=participant.campaign_id,
        participant_id=participant.id,
        payload={"discipline": participant.discipline, "institution_supplied": bool(participant.institution)},
    )
    db.commit()
    db.refresh(participant)
    return SessionProfileResponse(
        name=participant.name,
        institution=participant.institution,
        discipline=participant.discipline,
        discipline_other=participant.discipline_other,
        profile_completed=True,
    )


@router.post("/block-b-comprehension", response_model=BlockBComprehensionResponse)
def submit_block_b_comprehension(
    payload: BlockBComprehensionRequest,
    db: Session = Depends(get_db),
    participant: Participant = Depends(get_current_participant),
):
    campaign = _campaign_or_404(db, participant.campaign_id)
    protocol_version = campaign.protocol_version or get_settings().expert_protocol_version
    progress_data = compute_progress(db, campaign_id=participant.campaign_id, participant_id=participant.id)
    if progress_data["block_a_total"] > 0 and progress_data["block_a_completed"] < progress_data["block_a_total"]:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Block A must be completed before Block B.")
    if progress_data["block_a_total"] > 0 and not progress_data["block_a_feedback_completed"]:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Block A feedback must be completed before Block B.")

    selected_option = payload.selected_option.strip()
    passed = selected_option == BLOCK_B_COMPREHENSION_CORRECT_OPTION
    if passed:
        participant.block_b_comprehension_passed_at = participant.block_b_comprehension_passed_at or datetime.now(timezone.utc)
    else:
        participant.block_b_comprehension_attempts += 1
        if participant.block_b_comprehension_attempts >= BLOCK_B_COMPREHENSION_MAX_ATTEMPTS:
            participant.comprehension_risk = True

    log_event(
        db,
        action="session.block_b_comprehension",
        campaign_id=participant.campaign_id,
        participant_id=participant.id,
        payload={
            "selected_option": selected_option,
            "passed": passed,
            "attempts": participant.block_b_comprehension_attempts,
            "comprehension_risk": participant.comprehension_risk,
            "protocol_version": protocol_version,
        },
    )
    db.commit()
    db.refresh(participant)
    return BlockBComprehensionResponse(
        passed=participant.block_b_comprehension_passed_at is not None,
        attempts=participant.block_b_comprehension_attempts,
        max_attempts=BLOCK_B_COMPREHENSION_MAX_ATTEMPTS,
        comprehension_risk=participant.comprehension_risk,
        protocol_version=protocol_version,
    )


@router.get("/feedback/{block}", response_model=StageFeedbackResponse)
def get_session_feedback(
    block: str,
    db: Session = Depends(get_db),
    participant: Participant = Depends(get_current_participant),
):
    normalized_block = block.upper()
    if normalized_block not in {"A", "B", "C"}:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Unknown block.")
    feedback = get_stage_feedback(db, participant=participant, block=normalized_block)
    return StageFeedbackResponse(
        block=normalized_block,
        comment=feedback.comment if feedback else None,
        completed=feedback is not None,
    )


@router.put("/feedback/{block}", response_model=StageFeedbackResponse)
def put_session_feedback(
    block: str,
    payload: StageFeedbackUpdateRequest,
    db: Session = Depends(get_db),
    participant: Participant = Depends(get_current_participant),
):
    normalized_block = block.upper()
    if normalized_block not in {"A", "B", "C"}:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Unknown block.")

    progress_data = compute_progress(db, campaign_id=participant.campaign_id, participant_id=participant.id)
    comment = payload.comment.strip()
    if len(comment) < 12:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Stage feedback must be at least 12 characters.",
        )
    if normalized_block == "A" and progress_data["block_a_total"] > 0 and progress_data["block_a_completed"] < progress_data["block_a_total"]:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Block A must be completed first.")
    if normalized_block == "B":
        if progress_data["block_b_completed"] < progress_data["block_b_total"]:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Block B must be completed first.")
        if progress_data["block_a_total"] > 0 and not progress_data["block_a_feedback_completed"]:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Block A feedback must be completed first.")
    if normalized_block == "C":
        if progress_data.get("block_c_total", 0) <= 0:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Block C is not enabled for this campaign.")
        if progress_data["block_b_total"] > 0 and progress_data["block_b_completed"] < progress_data["block_b_total"]:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Block B must be completed first.")
        if progress_data["block_b_total"] > 0 and not progress_data["block_b_feedback_completed"]:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Block B feedback must be completed first.")
        if progress_data.get("block_c_completed", 0) < progress_data.get("block_c_total", 0):
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Block C must be completed first.")

    feedback = upsert_stage_feedback(
        db,
        participant=participant,
        block=normalized_block,
        comment=comment,
    )
    log_event(
        db,
        action="session.feedback.submit",
        campaign_id=participant.campaign_id,
        participant_id=participant.id,
        payload={"block": normalized_block},
    )
    db.commit()
    return StageFeedbackResponse(block=normalized_block, comment=feedback.comment, completed=True)
