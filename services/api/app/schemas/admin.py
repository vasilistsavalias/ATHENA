from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class ImportPackRequest(BaseModel):
    pack_dir: str = Field(min_length=1)
    campaign_name: str = Field(min_length=1, max_length=255)
    seed: int = 42
    stage13_samples: str | None = None
    activate: bool = True
    disjoint_blocks: bool = True


class ImportPackResponse(BaseModel):
    campaign_id: int
    campaign_name: str
    is_active: bool


class AdminLoginRequest(BaseModel):
    password: str = Field(min_length=1, max_length=255)


class AdminSessionResponse(BaseModel):
    authenticated: bool
    auth_mode: str
    campaign_id: int | None = None
    campaign_name: str | None = None


class AdminRuntimeResetRequest(BaseModel):
    confirm_phrase: str = Field(min_length=1, max_length=255)
    campaign_id: int | None = Field(default=None, ge=1)
    all_active_campaigns: bool = False
    all_campaigns: bool = False
    remove_assets: bool = False


class AdminRuntimeResetResponse(BaseModel):
    campaign_ids: list[int]
    deleted_counts: dict[str, int]
    removed_asset_dirs: list[str]
    next_participant_public_id: str
    identity_reset: bool
    timestamp: datetime


class AdminStatsResponse(BaseModel):
    participants_total: int
    participants_completed: int
    participants_active: int
    profiles_completed: int
    block_a_feedback_completed: int
    block_b_feedback_completed: int
    block_c_feedback_completed: int
    block_a_responses: int
    block_b_responses: int
    block_c_responses: int
    attention_flags_total: int
    comprehension_risk_total: int


class DisciplineBreakdownEntry(BaseModel):
    discipline: str
    count: int


class AdminParticipantRow(BaseModel):
    participant_id: str
    status: str
    name: str | None = None
    institution: str | None = None
    discipline: str | None = None
    profile_completed: bool
    block_b_comprehension_attempts: int
    block_b_comprehension_passed: bool
    comprehension_risk: bool
    block_a_completed: int
    block_a_total: int
    block_b_completed: int
    block_b_total: int
    block_c_completed: int
    block_c_total: int
    block_a_feedback_completed: bool
    block_b_feedback_completed: bool
    block_c_feedback_completed: bool
    attention_flags: int
    block_a_stage_comment: str | None = None
    block_b_stage_comment: str | None = None
    block_c_stage_comment: str | None = None
    created_at: datetime
    completed_at: datetime | None = None


class AdminCommentRow(BaseModel):
    participant_id: str
    block: str
    source: str
    comment: str
    sample_id: str | None = None
    created_at: datetime


class AdminCampaignSummary(BaseModel):
    id: int
    name: str
    seed: int
    protocol_version: str
    block_a_target_count: int
    block_b_target_count: int


class AdminDashboardResponse(BaseModel):
    campaign: AdminCampaignSummary | None = None
    stats: AdminStatsResponse
    discipline_breakdown: list[DisciplineBreakdownEntry]
    participants: list[AdminParticipantRow]
    recent_stage_feedback: list[AdminCommentRow]
    recent_item_comments: list[AdminCommentRow]
