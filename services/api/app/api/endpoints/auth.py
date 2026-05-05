from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.security import create_session_token, decode_session_token, secure_compare
from app.db.session import get_db
from app.models import BlockBItem, Campaign, Participant
from app.schemas.auth import InviteRequest, InviteResponse, LoginRequest, LoginResponse
from app.schemas.session import CampaignInfo, ProgressResponse
from app.services.assignment_service import compute_progress, ensure_assignments
from app.services.audit_service import log_event

router = APIRouter()


def _active_campaign(db: Session) -> Campaign:
    campaign = db.query(Campaign).filter(Campaign.is_active.is_(True)).order_by(Campaign.id.desc()).first()
    if campaign is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No active campaign.")
    return campaign


def _block_c_target_count(db: Session, campaign_id: int) -> int:
    total = 0
    for row in db.query(BlockBItem).filter(BlockBItem.campaign_id == campaign_id).all():
        part = str((row.metadata_json or {}).get("block_part") or "B").upper()
        if part == "C":
            total += 1
    return total


def _campaign_info(db: Session, campaign: Campaign, *, block_c_target_count: int | None = None) -> CampaignInfo:
    settings = get_settings()
    block_c = _block_c_target_count(db, campaign.id) if block_c_target_count is None else int(block_c_target_count)
    return CampaignInfo(
        id=campaign.id,
        name=campaign.name,
        seed=campaign.seed,
        protocol_version=campaign.protocol_version or settings.expert_protocol_version,
        block_a_target_count=campaign.block_a_target_count,
        block_b_target_count=campaign.block_b_target_count,
        block_c_target_count=block_c,
    )


def _set_session_cookie(*, response: Response, participant_id: int, campaign_id: int):
    settings = get_settings()
    token = create_session_token(participant_id, campaign_id)
    cookie_kwargs: dict = dict(
        key=settings.session_cookie_name,
        value=token,
        httponly=True,
        secure=settings.effective_cookie_secure,
        samesite=settings.effective_cookie_samesite,
        max_age=settings.session_ttl_hours * 3600,
    )
    if settings.effective_cookie_domain:
        cookie_kwargs["domain"] = settings.effective_cookie_domain
    response.set_cookie(**cookie_kwargs)


def _find_session_participant(*, request: Request, db: Session, campaign_id: int) -> Participant | None:
    settings = get_settings()
    token_value = request.cookies.get(settings.session_cookie_name)
    if not token_value:
        return None
    payload = decode_session_token(token_value)
    if not payload:
        return None
    participant = (
        db.query(Participant)
        .filter(
            Participant.id == int(payload.get("participant_id", -1)),
            Participant.campaign_id == int(payload.get("campaign_id", -1)),
        )
        .first()
    )
    if participant is None or participant.campaign_id != campaign_id:
        return None
    return participant


def _create_participant(*, db: Session, campaign: Campaign, action: str) -> Participant:
    participant = Participant(campaign_id=campaign.id, public_id="pending", status="active")
    db.add(participant)
    db.flush()
    participant.public_id = f"R{participant.id:04d}"
    log_event(
        db,
        action=action,
        campaign_id=campaign.id,
        participant_id=participant.id,
        payload={"participant_public_id": participant.public_id},
    )
    db.commit()
    db.refresh(participant)
    return participant


def _build_invite_response(*, db: Session, participant: Participant, campaign: Campaign) -> InviteResponse:
    ensure_assignments(db, campaign=campaign, participant=participant)
    progress_data = compute_progress(db, campaign_id=campaign.id, participant_id=participant.id)
    return InviteResponse(
        participant_public_id=participant.public_id,
        campaign=_campaign_info(db, campaign, block_c_target_count=progress_data.get("block_c_total", 0)),
        progress=ProgressResponse(**progress_data),
    )


@router.post("/invite", response_model=InviteResponse)
def invite(payload: InviteRequest, request: Request, response: Response, db: Session = Depends(get_db)):
    settings = get_settings()
    if not secure_compare(payload.invite_code, settings.app_invite_code):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid invite code.")

    campaign = _active_campaign(db)
    participant = _find_session_participant(request=request, db=db, campaign_id=campaign.id)
    if participant is None:
        participant = _create_participant(db=db, campaign=campaign, action="auth.invite")
    else:
        log_event(
            db,
            action="auth.invite.reuse",
            campaign_id=campaign.id,
            participant_id=participant.id,
            payload={"participant_public_id": participant.public_id},
        )
        db.commit()
    _set_session_cookie(response=response, participant_id=participant.id, campaign_id=campaign.id)
    return _build_invite_response(db=db, participant=participant, campaign=campaign)


@router.post("/login", response_model=LoginResponse)
def login(payload: LoginRequest, response: Response, db: Session = Depends(get_db)):
    settings = get_settings()
    username_ok = secure_compare(payload.username, settings.app_shared_username)
    password_ok = secure_compare(payload.password, settings.app_shared_password)
    if not username_ok or not password_ok:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials.")

    campaign = _active_campaign(db)
    participant = _create_participant(db=db, campaign=campaign, action="auth.login")
    _set_session_cookie(response=response, participant_id=participant.id, campaign_id=campaign.id)

    return LoginResponse(
        participant_public_id=participant.public_id,
        campaign=_campaign_info(db, campaign),
    )


@router.post("/logout")
def logout(response: Response):
    settings = get_settings()
    delete_kwargs: dict = dict(
        key=settings.session_cookie_name,
        secure=settings.effective_cookie_secure,
        samesite=settings.session_cookie_samesite,
        httponly=True,
        path="/",
    )
    if settings.effective_cookie_domain:
        delete_kwargs["domain"] = settings.effective_cookie_domain
    response.delete_cookie(**delete_kwargs)
    return {"message": "Logged out"}
