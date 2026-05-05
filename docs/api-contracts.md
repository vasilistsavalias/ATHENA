# API Contracts

Canonical sources:

- `services/api/app/api/endpoints/`
- `services/web/src/lib/api.ts`

## Participant endpoints

- `POST /api/v1/auth/invite`
- `POST /api/v1/auth/logout`
- `GET /api/v1/session/me`
- `GET /api/v1/session/profile`
- `PUT /api/v1/session/profile`
- `GET /api/v1/session/feedback/A`
- `PUT /api/v1/session/feedback/A`
- `GET /api/v1/session/feedback/B`
- `PUT /api/v1/session/feedback/B`
- `GET /api/v1/campaign/active`
- `GET /api/v1/block-a/next`
- `POST /api/v1/block-a/submit`
- `GET /api/v1/block-b/next`
- `POST /api/v1/block-b/submit`
- `GET /api/v1/progress`
- `POST /api/v1/session/complete`

## Admin endpoints

- `POST /api/v1/admin/auth/login`
- `POST /api/v1/admin/auth/logout`
- `GET /api/v1/admin/session`
- `GET /api/v1/admin/dashboard`
- `POST /api/v1/admin/import-pack`
- `GET /api/v1/admin/export/responses.csv`
- `GET /api/v1/admin/export/responses.json`
- `GET /api/v1/admin/export/quality_report.json`

## Contract notes

- invite returns participant context and campaign context
- assignment endpoints return the next eligible item, not arbitrary browsing
- submit endpoints enforce one response per assignment
- `POST /api/v1/block-a/submit` requires `comment` with `comment.strip().length > 0`
- `POST /api/v1/block-b/submit` requires `comment` with `comment.strip().length > 0`
- admin routes require cookie-backed admin auth or `x-admin-secret`
