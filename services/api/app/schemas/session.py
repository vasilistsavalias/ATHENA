from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel


class CampaignInfo(BaseModel):
    id: int
    name: str
    seed: int
    protocol_version: str
    block_a_target_count: int
    block_b_target_count: int
    block_c_target_count: int = 0


class SessionInfo(BaseModel):
    participant_id: str
    status: str
    profile_completed: bool
    block_b_comprehension_attempts: int
    block_b_comprehension_passed: bool
    comprehension_risk: bool
    campaign: CampaignInfo
    created_at: datetime
    completed_at: datetime | None = None


class ProgressResponse(BaseModel):
    block_a_completed: int
    block_a_total: int
    block_b_completed: int
    block_b_total: int
    block_c_completed: int
    block_c_total: int
    profile_completed: bool
    block_a_feedback_completed: bool
    block_b_feedback_completed: bool
    block_c_feedback_completed: bool
    is_complete: bool


class SessionProfileResponse(BaseModel):
    name: str | None = None
    institution: str | None = None
    discipline: str | None = None
    discipline_other: str | None = None
    profile_completed: bool


class SessionProfileUpdateRequest(BaseModel):
    name: str | None = None
    institution: str | None = None
    discipline: str | None = None
    discipline_other: str | None = None


class StageFeedbackResponse(BaseModel):
    block: str
    comment: str | None = None
    completed: bool


class StageFeedbackUpdateRequest(BaseModel):
    comment: str


class BlockBComprehensionRequest(BaseModel):
    selected_option: str


class BlockBComprehensionResponse(BaseModel):
    passed: bool
    attempts: int
    max_attempts: int
    comprehension_risk: bool
    protocol_version: str
