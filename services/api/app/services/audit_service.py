from __future__ import annotations

from sqlalchemy.orm import Session

from app.models import AuditEvent


def log_event(
    db: Session,
    *,
    action: str,
    campaign_id: int | None = None,
    participant_id: int | None = None,
    payload: dict | None = None,
):
    db.add(
        AuditEvent(
            campaign_id=campaign_id,
            participant_id=participant_id,
            action=action,
            payload_json=payload or {},
        )
    )

