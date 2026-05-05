from __future__ import annotations

from pydantic import BaseModel, Field

from app.schemas.session import CampaignInfo, ProgressResponse


class LoginRequest(BaseModel):
    username: str = Field(min_length=1, max_length=128)
    password: str = Field(min_length=1, max_length=256)


class LoginResponse(BaseModel):
    participant_public_id: str
    campaign: CampaignInfo


class InviteRequest(BaseModel):
    invite_code: str = Field(min_length=1, max_length=256)


class InviteResponse(BaseModel):
    participant_public_id: str
    campaign: CampaignInfo
    progress: ProgressResponse
