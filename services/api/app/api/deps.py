from __future__ import annotations

from fastapi import Depends, Header, HTTPException, Request, status
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.security import decode_admin_session_token, decode_session_token, secure_compare
from app.db.session import get_db
from app.models import Campaign, Participant


def get_active_campaign(db: Session = Depends(get_db)) -> Campaign:
    campaign = db.query(Campaign).filter(Campaign.is_active.is_(True)).order_by(Campaign.id.desc()).first()
    if campaign is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No active campaign found.")
    return campaign


def get_current_participant(
    request: Request,
    db: Session = Depends(get_db),
) -> Participant:
    settings = get_settings()
    token_value = request.cookies.get(settings.session_cookie_name)
    if not token_value:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated.")
    payload = decode_session_token(token_value)
    if not payload:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid session token.")
    participant = (
        db.query(Participant)
        .filter(
            Participant.id == int(payload.get("participant_id", -1)),
            Participant.campaign_id == int(payload.get("campaign_id", -1)),
        )
        .first()
    )
    if participant is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Participant session not found.")
    if participant.campaign is None or not participant.campaign.is_active:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Participant session is stale.")
    return participant


def require_admin_secret(x_admin_secret: str = Header(default="")):
    settings = get_settings()
    if not secure_compare(x_admin_secret, settings.admin_export_secret):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid admin secret.")
    return True


def require_admin_access(request: Request, x_admin_secret: str = Header(default="")):
    settings = get_settings()
    if x_admin_secret and secure_compare(x_admin_secret, settings.admin_export_secret):
        return {"auth_mode": "header"}

    token_value = request.cookies.get(settings.admin_cookie_name)
    if token_value and decode_admin_session_token(token_value):
        return {"auth_mode": "cookie"}

    raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Admin authentication required.")
