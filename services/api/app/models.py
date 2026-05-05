from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import (
    JSON,
    Boolean,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base


def _utcnow():
    return datetime.now(timezone.utc)


class Campaign(Base):
    __tablename__ = "campaigns"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    seed: Mapped[int] = mapped_column(Integer, nullable=False, default=42)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    protocol_version: Mapped[str] = mapped_column(String(64), nullable=False, default="ATHENA Expert Protocol v1.1")
    block_a_target_count: Mapped[int] = mapped_column(Integer, nullable=False, default=25)
    block_b_target_count: Mapped[int] = mapped_column(Integer, nullable=False, default=15)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=_utcnow)

    participants: Mapped[list["Participant"]] = relationship(back_populates="campaign", cascade="all, delete-orphan")


class Participant(Base):
    __tablename__ = "participants"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    campaign_id: Mapped[int] = mapped_column(ForeignKey("campaigns.id", ondelete="CASCADE"), nullable=False)
    public_id: Mapped[str] = mapped_column(String(32), nullable=False, unique=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="active")
    name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    institution: Mapped[str | None] = mapped_column(String(255), nullable=True)
    discipline: Mapped[str | None] = mapped_column(String(64), nullable=True)
    discipline_other: Mapped[str | None] = mapped_column(String(255), nullable=True)
    profile_completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    block_b_comprehension_attempts: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    block_b_comprehension_passed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    comprehension_risk: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=_utcnow)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    campaign: Mapped["Campaign"] = relationship(back_populates="participants")


class BlockAItem(Base):
    __tablename__ = "block_a_items"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    campaign_id: Mapped[int] = mapped_column(ForeignKey("campaigns.id", ondelete="CASCADE"), nullable=False)
    sample_id: Mapped[str] = mapped_column(String(255), nullable=False)
    image_url: Mapped[str] = mapped_column(Text, nullable=False)
    source_label: Mapped[str] = mapped_column(String(64), nullable=False, default="generated")
    mask_type: Mapped[str | None] = mapped_column(String(64), nullable=True)
    mask_coverage_bin: Mapped[str | None] = mapped_column(String(64), nullable=True)
    metadata_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)


class BlockBItem(Base):
    __tablename__ = "block_b_items"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    campaign_id: Mapped[int] = mapped_column(ForeignKey("campaigns.id", ondelete="CASCADE"), nullable=False)
    sample_id: Mapped[str] = mapped_column(String(255), nullable=False)
    input_url: Mapped[str] = mapped_column(Text, nullable=False)
    option_a_url: Mapped[str] = mapped_column(Text, nullable=False)
    option_b_url: Mapped[str] = mapped_column(Text, nullable=False)
    mask_type: Mapped[str | None] = mapped_column(String(64), nullable=True)
    mask_coverage_bin: Mapped[str | None] = mapped_column(String(64), nullable=True)
    metadata_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)


class BlockAAssignment(Base):
    __tablename__ = "block_a_assignments"
    __table_args__ = (
        UniqueConstraint("campaign_id", "participant_id", "item_order", name="uq_block_a_order"),
        Index("ix_block_a_assignment_lookup", "campaign_id", "participant_id", "item_order"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    campaign_id: Mapped[int] = mapped_column(ForeignKey("campaigns.id", ondelete="CASCADE"), nullable=False)
    participant_id: Mapped[int] = mapped_column(ForeignKey("participants.id", ondelete="CASCADE"), nullable=False)
    item_id: Mapped[int] = mapped_column(ForeignKey("block_a_items.id", ondelete="CASCADE"), nullable=False)
    item_order: Mapped[int] = mapped_column(Integer, nullable=False)
    is_attention_check: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    attention_source_item_id: Mapped[int | None] = mapped_column(
        ForeignKey("block_a_items.id", ondelete="SET NULL"), nullable=True
    )
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    item: Mapped["BlockAItem"] = relationship(foreign_keys=[item_id])


class BlockBAssignment(Base):
    __tablename__ = "block_b_assignments"
    __table_args__ = (
        UniqueConstraint("campaign_id", "participant_id", "item_order", name="uq_block_b_order"),
        Index("ix_block_b_assignment_lookup", "campaign_id", "participant_id", "item_order"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    campaign_id: Mapped[int] = mapped_column(ForeignKey("campaigns.id", ondelete="CASCADE"), nullable=False)
    participant_id: Mapped[int] = mapped_column(ForeignKey("participants.id", ondelete="CASCADE"), nullable=False)
    item_id: Mapped[int] = mapped_column(ForeignKey("block_b_items.id", ondelete="CASCADE"), nullable=False)
    item_order: Mapped[int] = mapped_column(Integer, nullable=False)
    show_a_left: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    is_attention_check: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    attention_source_item_id: Mapped[int | None] = mapped_column(
        ForeignKey("block_b_items.id", ondelete="SET NULL"), nullable=True
    )
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    item: Mapped["BlockBItem"] = relationship(foreign_keys=[item_id])


class BlockAResponse(Base):
    __tablename__ = "block_a_responses"
    __table_args__ = (
        UniqueConstraint("assignment_id", name="uq_block_a_response_assignment"),
        Index("ix_block_a_response_campaign_participant", "campaign_id", "participant_id"),
        CheckConstraint("authenticity_likelihood BETWEEN 1 AND 5", name="ck_block_a_authenticity_range"),
        CheckConstraint("archaeological_plausibility BETWEEN 1 AND 5", name="ck_block_a_plausibility_range"),
        CheckConstraint("confidence BETWEEN 1 AND 5", name="ck_block_a_confidence_range"),
        CheckConstraint("response_time_ms >= 0", name="ck_block_a_response_time_nonnegative"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    campaign_id: Mapped[int] = mapped_column(ForeignKey("campaigns.id", ondelete="CASCADE"), nullable=False)
    participant_id: Mapped[int] = mapped_column(ForeignKey("participants.id", ondelete="CASCADE"), nullable=False)
    assignment_id: Mapped[int] = mapped_column(ForeignKey("block_a_assignments.id", ondelete="CASCADE"), nullable=False)
    authenticity_likelihood: Mapped[int] = mapped_column(Integer, nullable=False)
    archaeological_plausibility: Mapped[int] = mapped_column(Integer, nullable=False)
    confidence: Mapped[int] = mapped_column(Integer, nullable=False)
    comment: Mapped[str | None] = mapped_column(Text, nullable=True)
    response_time_ms: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=_utcnow)


class BlockBResponse(Base):
    __tablename__ = "block_b_responses"
    __table_args__ = (
        UniqueConstraint("assignment_id", name="uq_block_b_response_assignment"),
        Index("ix_block_b_response_campaign_participant", "campaign_id", "participant_id"),
        CheckConstraint("confidence BETWEEN 1 AND 5", name="ck_block_b_confidence_range"),
        CheckConstraint("response_time_ms >= 0", name="ck_block_b_response_time_nonnegative"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    campaign_id: Mapped[int] = mapped_column(ForeignKey("campaigns.id", ondelete="CASCADE"), nullable=False)
    participant_id: Mapped[int] = mapped_column(ForeignKey("participants.id", ondelete="CASCADE"), nullable=False)
    assignment_id: Mapped[int] = mapped_column(ForeignKey("block_b_assignments.id", ondelete="CASCADE"), nullable=False)
    choice: Mapped[str] = mapped_column(String(16), nullable=False)
    confidence: Mapped[int] = mapped_column(Integer, nullable=False)
    comment: Mapped[str | None] = mapped_column(Text, nullable=True)
    response_time_ms: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=_utcnow)


class AttentionFlag(Base):
    __tablename__ = "attention_flags"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    campaign_id: Mapped[int] = mapped_column(ForeignKey("campaigns.id", ondelete="CASCADE"), nullable=False)
    participant_id: Mapped[int] = mapped_column(ForeignKey("participants.id", ondelete="CASCADE"), nullable=False)
    block: Mapped[str] = mapped_column(String(8), nullable=False)
    assignment_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    flag_type: Mapped[str] = mapped_column(String(64), nullable=False)
    details_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=_utcnow)


class AuditEvent(Base):
    __tablename__ = "audit_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    campaign_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    participant_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    action: Mapped[str] = mapped_column(String(128), nullable=False)
    payload_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=_utcnow)


class StageFeedback(Base):
    __tablename__ = "stage_feedback"
    __table_args__ = (
        UniqueConstraint("campaign_id", "participant_id", "block", name="uq_stage_feedback_block"),
        CheckConstraint("block IN ('A', 'B', 'C')", name="ck_stage_feedback_block"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    campaign_id: Mapped[int] = mapped_column(ForeignKey("campaigns.id", ondelete="CASCADE"), nullable=False)
    participant_id: Mapped[int] = mapped_column(ForeignKey("participants.id", ondelete="CASCADE"), nullable=False)
    block: Mapped[str] = mapped_column(String(8), nullable=False)
    comment: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=_utcnow, onupdate=_utcnow)
