from __future__ import annotations

from datetime import timedelta
import hmac

from itsdangerous import BadSignature, BadTimeSignature, URLSafeTimedSerializer

from app.core.config import get_settings


def _serializer(salt: str) -> URLSafeTimedSerializer:
    settings = get_settings()
    return URLSafeTimedSerializer(settings.session_secret, salt=salt)


def create_session_token(participant_id: int, campaign_id: int) -> str:
    payload = {"participant_id": int(participant_id), "campaign_id": int(campaign_id)}
    return _serializer("archaeologist-eval-session").dumps(payload)


def decode_session_token(token: str):
    settings = get_settings()
    max_age = int(timedelta(hours=settings.session_ttl_hours).total_seconds())
    try:
        return _serializer("archaeologist-eval-session").loads(token, max_age=max_age)
    except (BadSignature, BadTimeSignature):
        return None


def create_admin_session_token() -> str:
    payload = {"role": "admin"}
    return _serializer("archaeologist-eval-admin").dumps(payload)


def decode_admin_session_token(token: str):
    settings = get_settings()
    max_age = int(timedelta(hours=settings.admin_session_ttl_hours).total_seconds())
    try:
        payload = _serializer("archaeologist-eval-admin").loads(token, max_age=max_age)
        if payload.get("role") != "admin":
            return None
        return payload
    except (BadSignature, BadTimeSignature):
        return None


def secure_compare(left: str, right: str) -> bool:
    return hmac.compare_digest(str(left), str(right))
