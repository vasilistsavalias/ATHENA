from __future__ import annotations

from sqlalchemy.orm import Session

from app.models import Participant, StageFeedback


def get_stage_feedback(
    db: Session,
    *,
    participant: Participant,
    block: str,
) -> StageFeedback | None:
    return (
        db.query(StageFeedback)
        .filter(
            StageFeedback.campaign_id == participant.campaign_id,
            StageFeedback.participant_id == participant.id,
            StageFeedback.block == block,
        )
        .first()
    )


def upsert_stage_feedback(
    db: Session,
    *,
    participant: Participant,
    block: str,
    comment: str,
) -> StageFeedback:
    feedback = get_stage_feedback(db, participant=participant, block=block)
    if feedback is None:
        feedback = StageFeedback(
            campaign_id=participant.campaign_id,
            participant_id=participant.id,
            block=block,
            comment=comment,
        )
        db.add(feedback)
    else:
        feedback.comment = comment
    db.flush()
    return feedback
