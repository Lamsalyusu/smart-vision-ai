"""baseline schema

The schema as it was before migrations were introduced, when tables were made
with ``Base.metadata.create_all``. Tables and indexes that already exist are
skipped, so databases created that way upgrade without being recreated.

Revision ID: 0001_baseline
Revises:
Create Date: 2026-11-05 21:30:57.956195
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op


revision: str = '0001_baseline'
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _existing_tables() -> set[str]:
    return set(sa.inspect(op.get_bind()).get_table_names())


def _create_table(name: str, *columns, **kwargs) -> None:
    if name not in _existing_tables():
        op.create_table(name, *columns, **kwargs)


def _create_index(name: str, table: str, columns: list[str]) -> None:
    existing = {index["name"] for index in sa.inspect(op.get_bind()).get_indexes(table)}
    if name not in existing:
        op.create_index(name, table, columns, unique=False)


def upgrade() -> None:
    _create_table('audit_logs',
    sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
    sa.Column('event_type', sa.String(), nullable=False),
    sa.Column('user_id', sa.String(), nullable=True),
    sa.Column('details', sa.JSON(), nullable=False),
    sa.Column('timestamp', sa.DateTime(), nullable=False),
    sa.PrimaryKeyConstraint('id')
    )
    _create_table('crops',
    sa.Column('id', sa.String(), nullable=False),
    sa.Column('object_class', sa.String(), nullable=False),
    sa.Column('confidence', sa.Float(), nullable=False),
    sa.Column('timestamp', sa.DateTime(), nullable=False),
    sa.Column('distance_m', sa.Float(), nullable=True),
    sa.Column('has_crop', sa.Boolean(), nullable=False),
    sa.Column('payload', sa.JSON(), nullable=True),
    sa.PrimaryKeyConstraint('id')
    )
    _create_table('failed_login_attempts',
    sa.Column('identifier', sa.String(), nullable=False),
    sa.Column('attempt_count', sa.Integer(), nullable=False),
    sa.Column('locked_until', sa.DateTime(), nullable=True),
    sa.Column('updated_at', sa.DateTime(), nullable=False),
    sa.PrimaryKeyConstraint('identifier')
    )
    _create_table('replay_frames',
    sa.Column('id', sa.String(), nullable=False),
    sa.Column('session_id', sa.String(), nullable=False),
    sa.Column('captured_at', sa.Float(), nullable=False),
    sa.Column('detections', sa.JSON(), nullable=False),
    sa.Column('recorded_at', sa.DateTime(), nullable=False),
    sa.PrimaryKeyConstraint('id')
    )
    _create_index('ix_replay_session_captured', 'replay_frames', ['session_id', 'captured_at'])

    _create_table('telemetry',
    sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
    sa.Column('key', sa.String(), nullable=False),
    sa.Column('value', sa.JSON(), nullable=False),
    sa.Column('recorded_at', sa.DateTime(), nullable=False),
    sa.PrimaryKeyConstraint('id')
    )
    _create_table('users',
    sa.Column('id', sa.String(), nullable=False),
    sa.Column('full_name', sa.String(), nullable=False),
    sa.Column('email', sa.String(), nullable=False),
    sa.Column('username', sa.String(), nullable=False),
    sa.Column('password_hash', sa.String(), nullable=False),
    sa.Column('auth_provider', sa.String(), nullable=False),
    sa.Column('role', sa.String(), nullable=False),
    sa.Column('created_at', sa.DateTime(), nullable=False),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('email'),
    sa.UniqueConstraint('username')
    )
    _create_table('password_reset_tokens',
    sa.Column('token_hash', sa.String(), nullable=False),
    sa.Column('user_id', sa.String(), nullable=False),
    sa.Column('created_at', sa.DateTime(), nullable=False),
    sa.Column('expires_at', sa.DateTime(), nullable=False),
    sa.Column('used_at', sa.DateTime(), nullable=True),
    sa.ForeignKeyConstraint(['user_id'], ['users.id'], ),
    sa.PrimaryKeyConstraint('token_hash')
    )
    _create_index('ix_password_reset_tokens_user_id', 'password_reset_tokens', ['user_id'])

    _create_table('sessions',
    sa.Column('token', sa.String(), nullable=False),
    sa.Column('user_id', sa.String(), nullable=False),
    sa.Column('created_at', sa.DateTime(), nullable=False),
    sa.Column('last_active_at', sa.DateTime(), nullable=False),
    sa.Column('expires_at', sa.DateTime(), nullable=True),
    sa.ForeignKeyConstraint(['user_id'], ['users.id'], ),
    sa.PrimaryKeyConstraint('token')
    )


def downgrade() -> None:
    op.drop_table('sessions')
    with op.batch_alter_table('password_reset_tokens', schema=None) as batch_op:
        batch_op.drop_index(batch_op.f('ix_password_reset_tokens_user_id'))

    op.drop_table('password_reset_tokens')
    op.drop_table('users')
    op.drop_table('telemetry')
    with op.batch_alter_table('replay_frames', schema=None) as batch_op:
        batch_op.drop_index('ix_replay_session_captured')

    op.drop_table('replay_frames')
    op.drop_table('failed_login_attempts')
    op.drop_table('crops')
    op.drop_table('audit_logs')
