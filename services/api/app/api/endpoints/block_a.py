from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.api.deps import get_current_participant
from app.db.session import get_db
from app.models import BlockAAssignment, Participant
from app.schemas.block_a import BlockANextResponse, BlockAItemPayload, BlockASubmitRequest
from app.services.assignment_service import compute_progress, ensure_assignments, get_next_block_a_assignment, submit_block_a_response
from app.services.asset_url_service import normalize_block_a_image_url
from app.services.audit_service import log_event

router = APIRouter()


@router.get("/next", response_model=BlockANextResponse)
def next_block_a(
    db: Session = Depends(get_db),
    participant: Participant = Depends(get_current_participant),
):
    if participant.status == "completed":
        return BlockANextResponse(done=True, item=None)

    ensure_assignments(db, campaign=participant.campaign, participant=participant)
    progress_data = compute_progress(db, campaign_id=participant.campaign_id, participant_id=participant.id)
    if not progress_data["profile_completed"] and progress_data["block_a_completed"] == 0 and progress_data["block_b_completed"] == 0:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Profile must be completed before Block A.",
        )
    assignment = get_next_block_a_assignment(db, campaign_id=participant.campaign_id, participant_id=participant.id)
    if assignment is None:
        return BlockANextResponse(done=True, item=None)

    total = (
        db.query(BlockAAssignment)
        .filter(
            BlockAAssignment.campaign_id == participant.campaign_id,
            BlockAAssignment.participant_id == participant.id,
        )
        .count()
    )
    payload = BlockAItemPayload(
        assignment_id=assignment.id,
        item_order=assignment.item_order,
        total_items=total,
        sample_id=assignment.item.sample_id,
        image_url=normalize_block_a_image_url(
            campaign_id=participant.campaign_id,
            sample_id=assignment.item.sample_id,
            image_url=assignment.item.image_url,
            metadata_json=assignment.item.metadata_json,
        ),
        mask_type=assignment.item.mask_type,
        mask_coverage_bin=assignment.item.mask_coverage_bin,
        source_label=assignment.item.source_label,
        is_attention_check=assignment.is_attention_check,
    )
    return BlockANextResponse(done=False, item=payload)


@router.post("/submit", response_model=BlockANextResponse)
def submit_block_a(
    payload: BlockASubmitRequest,
    db: Session = Depends(get_db),
    participant: Participant = Depends(get_current_participant),
):
    if participant.status == "completed":
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Session already completed.")
    progress_data = compute_progress(db, campaign_id=participant.campaign_id, participant_id=participant.id)
    if not progress_data["profile_completed"] and progress_data["block_a_completed"] == 0 and progress_data["block_b_completed"] == 0:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Profile must be completed before Block A.",
        )

    assignment = (
        db.query(BlockAAssignment)
        .filter(
            BlockAAssignment.id == payload.assignment_id,
            BlockAAssignment.participant_id == participant.id,
            BlockAAssignment.campaign_id == participant.campaign_id,
        )
        .first()
    )
    if assignment is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Assignment not found.")

    try:
        submit_block_a_response(
            db,
            assignment=assignment,
            participant=participant,
            authenticity_likelihood=payload.authenticity_likelihood,
            archaeological_plausibility=payload.archaeological_plausibility,
            confidence=payload.confidence,
            comment=payload.comment,
            response_time_ms=payload.response_time_ms,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc

    log_event(
        db,
        action="block_a.submit",
        campaign_id=participant.campaign_id,
        participant_id=participant.id,
        payload={"assignment_id": assignment.id, "item_order": assignment.item_order},
    )
    db.commit()
    return next_block_a(db=db, participant=participant)
