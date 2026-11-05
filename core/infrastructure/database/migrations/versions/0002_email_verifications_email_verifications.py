"""email verifications

Revision ID: 0002_email_verifications
Revises: 0001_baseline
Create Date: 2026-11-05 21:32:19.878092
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op


revision: str = '0002_email_verifications'
down_revision: str | None = '0001_baseline'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table('email_verifications',
    sa.Column('user_id', sa.String(), nullable=False),
    sa.Column('token_hash', sa.String(), nullable=True),
    sa.Column('created_at', sa.DateTime(), nullable=False),
    sa.Column('expires_at', sa.DateTime(), nullable=True),
    sa.Column('verified_at', sa.DateTime(), nullable=True),
    sa.ForeignKeyConstraint(['user_id'], ['users.id'], ),
    sa.PrimaryKeyConstraint('user_id'),
    sa.UniqueConstraint('token_hash')
    )


def downgrade() -> None:
    op.drop_table('email_verifications')
