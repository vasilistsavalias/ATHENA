from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.api.deps import get_current_participant
from app.db.session import get_db
from app.models import Participant
from app.schemas.session import ProgressResponse
from app.services.assignment_service import compute_progress, ensure_assignments

router = APIRouter()


@router.get("/progress", response_model=ProgressResponse)
def progress(
    db: Session = Depends(get_db),
    participant: Participant = Depends(get_current_participant),
):
    ensure_assignments(db, campaign=participant.campaign, participant=participant)
    progress_data = compute_progress(db, campaign_id=participant.campaign_id, participant_id=participant.id)
    return ProgressResponse(**progress_data)
