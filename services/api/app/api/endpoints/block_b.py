from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.api.deps import get_current_participant
from app.db.session import get_db
from app.models import BlockBAssignment, Participant
from app.schemas.block_b import BlockBItemPayload, BlockBNextResponse, BlockBSubmitRequest
from app.services.assignment_service import (
    compute_progress,
    ensure_assignments,
    get_next_block_assignment_by_part,
    submit_block_b_response,
)
from app.services.asset_url_service import normalize_block_b_image_url
from app.services.audit_service import log_event

router = APIRouter()


@router.get("/next", response_model=BlockBNextResponse)
def next_block_b(
    db: Session = Depends(get_db),
    participant: Participant = Depends(get_current_participant),
):
    if participant.status == "completed":
        return BlockBNextResponse(done=True, item=None)

    ensure_assignments(db, campaign=participant.campaign, participant=participant)
    progress_data = compute_progress(db, campaign_id=participant.campaign_id, participant_id=participant.id)
    if progress_data["block_a_total"] > 0 and progress_data["block_a_completed"] < progress_data["block_a_total"]:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Block A must be completed before Block B.",
        )
    if progress_data["block_a_total"] > 0 and not progress_data["block_a_feedback_completed"]:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Block A feedback must be completed before Block B.",
        )
    assignment = get_next_block_assignment_by_part(
        db,
        campaign_id=participant.campaign_id,
        participant_id=participant.id,
        part="B",
    )
    if assignment is None:
        return BlockBNextResponse(done=True, item=None)
    is_practice = bool((assignment.item.metadata_json or {}).get("is_practice", False))

    all_b_assignments = (
        db.query(BlockBAssignment)
        .filter(
            BlockBAssignment.campaign_id == participant.campaign_id,
            BlockBAssignment.participant_id == participant.id,
        )
        .order_by(BlockBAssignment.item_order.asc())
        .all()
    )
    b_assignments = [row for row in all_b_assignments if str((row.item.metadata_json or {}).get("block_part") or "B").upper() == "B"]
    total = len(b_assignments)
    item_order = next((index for index, row in enumerate(b_assignments, start=1) if row.id == assignment.id), assignment.item_order)
    payload = BlockBItemPayload(
        assignment_id=assignment.id,
        item_order=item_order,
        total_items=total,
        sample_id=assignment.item.sample_id,
        input_url=normalize_block_b_image_url(
            campaign_id=participant.campaign_id,
            sample_id=assignment.item.sample_id,
            image_url=assignment.item.input_url,
        ),
        option_a_url=normalize_block_b_image_url(
            campaign_id=participant.campaign_id,
            sample_id=assignment.item.sample_id,
            image_url=assignment.item.option_a_url,
        ),
        option_b_url=normalize_block_b_image_url(
            campaign_id=participant.campaign_id,
            sample_id=assignment.item.sample_id,
            image_url=assignment.item.option_b_url,
        ),
        show_a_left=assignment.show_a_left,
        mask_type=assignment.item.mask_type,
        mask_coverage_bin=assignment.item.mask_coverage_bin,
        is_practice=is_practice,
        is_anchor=bool((assignment.item.metadata_json or {}).get("is_anchor", False)),
        is_attention_check=assignment.is_attention_check,
    )
    return BlockBNextResponse(done=False, item=payload)


@router.post("/submit", response_model=BlockBNextResponse)
def submit_block_b(
    payload: BlockBSubmitRequest,
    db: Session = Depends(get_db),
    participant: Participant = Depends(get_current_participant),
):
    if participant.status == "completed":
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Session already completed.")

    progress_data = compute_progress(db, campaign_id=participant.campaign_id, participant_id=participant.id)
    if progress_data["block_a_total"] > 0 and progress_data["block_a_completed"] < progress_data["block_a_total"]:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Block A must be completed before Block B.",
        )
    if progress_data["block_a_total"] > 0 and not progress_data["block_a_feedback_completed"]:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Block A feedback must be completed before Block B.",
        )
    assignment = (
        db.query(BlockBAssignment)
        .filter(
            BlockBAssignment.id == payload.assignment_id,
            BlockBAssignment.participant_id == participant.id,
            BlockBAssignment.campaign_id == participant.campaign_id,
        )
        .first()
    )
    if assignment is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Assignment not found.")
    part = str((assignment.item.metadata_json or {}).get("block_part") or "B").upper()
    if part != "B":
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Assignment does not belong to Block B.")

    try:
        submit_block_b_response(
            db,
            assignment=assignment,
            participant=participant,
            choice=payload.choice,
            confidence=payload.confidence,
            comment=payload.comment,
            response_time_ms=payload.response_time_ms,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc

    log_event(
        db,
        action="block_b.submit",
        campaign_id=participant.campaign_id,
        participant_id=participant.id,
        payload={"assignment_id": assignment.id, "item_order": assignment.item_order, "choice": payload.choice},
    )
    db.commit()
    return next_block_b(db=db, participant=participant)
