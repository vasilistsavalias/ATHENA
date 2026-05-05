# Data Models

This page summarizes the important persisted shapes rather than reproducing ORM code.

Canonical sources:

- `website/services/api/app/models.py`
- `website/services/api/alembic/versions/`

## Pipeline-side persisted state

- filtered image directories
- processed and split datasets
- masks, captions, and prompt artifacts
- model checkpoints
- evaluation matrices and statistical outputs

## Website-side persisted state

- campaigns
- participants
- Block A items and assignments
- Block B items and assignments
- Block A responses
- Block B responses
- stage feedback rows
- attention flags
- audit log events

## Export shape

The admin export service emits:

- participant-level rows
- stage-feedback rows
- item-level response rows
- quality summary report
