"""participant profile and stage feedback

Revision ID: 0002_profile_and_stage_feedback
Revises: 0001_initial_schema
Create Date: 2026-03-11 00:00:00
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "0002_profile_and_stage_feedback"
down_revision: Union[str, Sequence[str], None] = "0001_initial_schema"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("participants", sa.Column("name", sa.String(length=255), nullable=True))
    op.add_column("participants", sa.Column("institution", sa.String(length=255), nullable=True))
    op.add_column("participants", sa.Column("discipline", sa.String(length=64), nullable=True))
    op.add_column("participants", sa.Column("discipline_other", sa.String(length=255), nullable=True))
    op.add_column("participants", sa.Column("profile_completed_at", sa.DateTime(timezone=True), nullable=True))

    op.create_table(
        "stage_feedback",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("campaign_id", sa.Integer(), sa.ForeignKey("campaigns.id", ondelete="CASCADE"), nullable=False),
        sa.Column("participant_id", sa.Integer(), sa.ForeignKey("participants.id", ondelete="CASCADE"), nullable=False),
        sa.Column("block", sa.String(length=8), nullable=False),
        sa.Column("comment", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint("block IN ('A', 'B')", name="ck_stage_feedback_block"),
        sa.UniqueConstraint("campaign_id", "participant_id", "block", name="uq_stage_feedback_block"),
    )


def downgrade() -> None:
    op.drop_table("stage_feedback")
    op.drop_column("participants", "profile_completed_at")
    op.drop_column("participants", "discipline_other")
    op.drop_column("participants", "discipline")
    op.drop_column("participants", "institution")
    op.drop_column("participants", "name")
