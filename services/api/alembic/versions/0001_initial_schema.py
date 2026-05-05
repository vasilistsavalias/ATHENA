"""initial schema

Revision ID: 0001_initial_schema
Revises:
Create Date: 2026-03-03 00:00:00
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "0001_initial_schema"
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "campaigns",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("seed", sa.Integer(), nullable=False, server_default="42"),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("block_a_target_count", sa.Integer(), nullable=False, server_default="25"),
        sa.Column("block_b_target_count", sa.Integer(), nullable=False, server_default="15"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )

    op.create_table(
        "participants",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("campaign_id", sa.Integer(), sa.ForeignKey("campaigns.id", ondelete="CASCADE"), nullable=False),
        sa.Column("public_id", sa.String(length=32), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.UniqueConstraint("public_id", name="uq_participants_public_id"),
    )

    op.create_table(
        "block_a_items",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("campaign_id", sa.Integer(), sa.ForeignKey("campaigns.id", ondelete="CASCADE"), nullable=False),
        sa.Column("sample_id", sa.String(length=255), nullable=False),
        sa.Column("image_url", sa.Text(), nullable=False),
        sa.Column("source_label", sa.String(length=64), nullable=False),
        sa.Column("mask_type", sa.String(length=64), nullable=True),
        sa.Column("mask_coverage_bin", sa.String(length=64), nullable=True),
        sa.Column("metadata_json", sa.JSON(), nullable=True),
    )

    op.create_table(
        "block_b_items",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("campaign_id", sa.Integer(), sa.ForeignKey("campaigns.id", ondelete="CASCADE"), nullable=False),
        sa.Column("sample_id", sa.String(length=255), nullable=False),
        sa.Column("input_url", sa.Text(), nullable=False),
        sa.Column("option_a_url", sa.Text(), nullable=False),
        sa.Column("option_b_url", sa.Text(), nullable=False),
        sa.Column("mask_type", sa.String(length=64), nullable=True),
        sa.Column("mask_coverage_bin", sa.String(length=64), nullable=True),
        sa.Column("metadata_json", sa.JSON(), nullable=True),
    )

    op.create_table(
        "block_a_assignments",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("campaign_id", sa.Integer(), sa.ForeignKey("campaigns.id", ondelete="CASCADE"), nullable=False),
        sa.Column("participant_id", sa.Integer(), sa.ForeignKey("participants.id", ondelete="CASCADE"), nullable=False),
        sa.Column("item_id", sa.Integer(), sa.ForeignKey("block_a_items.id", ondelete="CASCADE"), nullable=False),
        sa.Column("item_order", sa.Integer(), nullable=False),
        sa.Column("is_attention_check", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column(
            "attention_source_item_id",
            sa.Integer(),
            sa.ForeignKey("block_a_items.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.UniqueConstraint("campaign_id", "participant_id", "item_order", name="uq_block_a_order"),
    )
    op.create_index(
        "ix_block_a_assignment_lookup",
        "block_a_assignments",
        ["campaign_id", "participant_id", "item_order"],
        unique=False,
    )

    op.create_table(
        "block_b_assignments",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("campaign_id", sa.Integer(), sa.ForeignKey("campaigns.id", ondelete="CASCADE"), nullable=False),
        sa.Column("participant_id", sa.Integer(), sa.ForeignKey("participants.id", ondelete="CASCADE"), nullable=False),
        sa.Column("item_id", sa.Integer(), sa.ForeignKey("block_b_items.id", ondelete="CASCADE"), nullable=False),
        sa.Column("item_order", sa.Integer(), nullable=False),
        sa.Column("show_a_left", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("is_attention_check", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column(
            "attention_source_item_id",
            sa.Integer(),
            sa.ForeignKey("block_b_items.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.UniqueConstraint("campaign_id", "participant_id", "item_order", name="uq_block_b_order"),
    )
    op.create_index(
        "ix_block_b_assignment_lookup",
        "block_b_assignments",
        ["campaign_id", "participant_id", "item_order"],
        unique=False,
    )

    op.create_table(
        "block_a_responses",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("campaign_id", sa.Integer(), sa.ForeignKey("campaigns.id", ondelete="CASCADE"), nullable=False),
        sa.Column("participant_id", sa.Integer(), sa.ForeignKey("participants.id", ondelete="CASCADE"), nullable=False),
        sa.Column(
            "assignment_id", sa.Integer(), sa.ForeignKey("block_a_assignments.id", ondelete="CASCADE"), nullable=False
        ),
        sa.Column("authenticity_likelihood", sa.Integer(), nullable=False),
        sa.Column("archaeological_plausibility", sa.Integer(), nullable=False),
        sa.Column("confidence", sa.Integer(), nullable=False),
        sa.Column("comment", sa.Text(), nullable=True),
        sa.Column("response_time_ms", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint("authenticity_likelihood BETWEEN 1 AND 5", name="ck_block_a_authenticity_range"),
        sa.CheckConstraint("archaeological_plausibility BETWEEN 1 AND 5", name="ck_block_a_plausibility_range"),
        sa.CheckConstraint("confidence BETWEEN 1 AND 5", name="ck_block_a_confidence_range"),
        sa.CheckConstraint("response_time_ms >= 0", name="ck_block_a_response_time_nonnegative"),
        sa.UniqueConstraint("assignment_id", name="uq_block_a_response_assignment"),
    )
    op.create_index(
        "ix_block_a_response_campaign_participant",
        "block_a_responses",
        ["campaign_id", "participant_id"],
        unique=False,
    )

    op.create_table(
        "block_b_responses",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("campaign_id", sa.Integer(), sa.ForeignKey("campaigns.id", ondelete="CASCADE"), nullable=False),
        sa.Column("participant_id", sa.Integer(), sa.ForeignKey("participants.id", ondelete="CASCADE"), nullable=False),
        sa.Column(
            "assignment_id", sa.Integer(), sa.ForeignKey("block_b_assignments.id", ondelete="CASCADE"), nullable=False
        ),
        sa.Column("choice", sa.String(length=16), nullable=False),
        sa.Column("confidence", sa.Integer(), nullable=False),
        sa.Column("comment", sa.Text(), nullable=True),
        sa.Column("response_time_ms", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint("confidence BETWEEN 1 AND 5", name="ck_block_b_confidence_range"),
        sa.CheckConstraint("response_time_ms >= 0", name="ck_block_b_response_time_nonnegative"),
        sa.UniqueConstraint("assignment_id", name="uq_block_b_response_assignment"),
    )
    op.create_index(
        "ix_block_b_response_campaign_participant",
        "block_b_responses",
        ["campaign_id", "participant_id"],
        unique=False,
    )

    op.create_table(
        "attention_flags",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("campaign_id", sa.Integer(), sa.ForeignKey("campaigns.id", ondelete="CASCADE"), nullable=False),
        sa.Column("participant_id", sa.Integer(), sa.ForeignKey("participants.id", ondelete="CASCADE"), nullable=False),
        sa.Column("block", sa.String(length=8), nullable=False),
        sa.Column("assignment_id", sa.Integer(), nullable=True),
        sa.Column("flag_type", sa.String(length=64), nullable=False),
        sa.Column("details_json", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )

    op.create_table(
        "audit_events",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("campaign_id", sa.Integer(), nullable=True),
        sa.Column("participant_id", sa.Integer(), nullable=True),
        sa.Column("action", sa.String(length=128), nullable=False),
        sa.Column("payload_json", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )


def downgrade() -> None:
    op.drop_table("audit_events")
    op.drop_table("attention_flags")
    op.drop_index("ix_block_b_response_campaign_participant", table_name="block_b_responses")
    op.drop_table("block_b_responses")
    op.drop_index("ix_block_a_response_campaign_participant", table_name="block_a_responses")
    op.drop_table("block_a_responses")
    op.drop_index("ix_block_b_assignment_lookup", table_name="block_b_assignments")
    op.drop_table("block_b_assignments")
    op.drop_index("ix_block_a_assignment_lookup", table_name="block_a_assignments")
    op.drop_table("block_a_assignments")
    op.drop_table("block_b_items")
    op.drop_table("block_a_items")
    op.drop_table("participants")
    op.drop_table("campaigns")
