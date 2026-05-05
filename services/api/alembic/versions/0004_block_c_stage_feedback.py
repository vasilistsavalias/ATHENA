"""allow block c stage feedback

Revision ID: 0004_block_c_stage_feedback
Revises: 0003_protocol_and_comprehension
Create Date: 2026-04-06 00:00:00
"""

from typing import Sequence, Union

from alembic import op


revision: str = "0004_block_c_stage_feedback"
down_revision: Union[str, Sequence[str], None] = "0003_protocol_and_comprehension"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table("stage_feedback") as batch_op:
        batch_op.drop_constraint("ck_stage_feedback_block", type_="check")
        batch_op.create_check_constraint("ck_stage_feedback_block", "block IN ('A', 'B', 'C')")


def downgrade() -> None:
    with op.batch_alter_table("stage_feedback") as batch_op:
        batch_op.drop_constraint("ck_stage_feedback_block", type_="check")
        batch_op.create_check_constraint("ck_stage_feedback_block", "block IN ('A', 'B')")
