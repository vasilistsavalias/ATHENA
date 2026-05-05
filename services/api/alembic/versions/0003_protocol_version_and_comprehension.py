"""protocol version and comprehension state

Revision ID: 0003_protocol_and_comprehension
Revises: 0002_profile_and_stage_feedback
Create Date: 2026-03-23 00:00:00
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "0003_protocol_and_comprehension"
down_revision: Union[str, Sequence[str], None] = "0002_profile_and_stage_feedback"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "campaigns",
        sa.Column("protocol_version", sa.String(length=64), nullable=False, server_default="ATHENA Expert Protocol v1.0"),
    )
    op.add_column(
        "participants",
        sa.Column("block_b_comprehension_attempts", sa.Integer(), nullable=False, server_default="0"),
    )
    op.add_column(
        "participants",
        sa.Column("block_b_comprehension_passed_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column(
        "participants",
        sa.Column("comprehension_risk", sa.Boolean(), nullable=False, server_default=sa.false()),
    )

    op.alter_column("campaigns", "protocol_version", server_default=None)
    op.alter_column("participants", "block_b_comprehension_attempts", server_default=None)
    op.alter_column("participants", "comprehension_risk", server_default=None)


def downgrade() -> None:
    op.drop_column("participants", "comprehension_risk")
    op.drop_column("participants", "block_b_comprehension_passed_at")
    op.drop_column("participants", "block_b_comprehension_attempts")
    op.drop_column("campaigns", "protocol_version")
