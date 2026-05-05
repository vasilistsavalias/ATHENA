# Website Architecture

The website is a campaign-driven expert-study system.

## Service layout

- `services/api/` — FastAPI backend
- `services/web/` — Next.js frontend
- `docker-compose.yml` — local Postgres + API + web

## Main responsibilities

API:

- participant sessions
- deterministic assignment delivery
- response persistence
- stage feedback persistence
- admin dashboard and exports
- pack import

Web:

- participant wizard flow
- admin login and dashboard
- server-side rewrite proxy to `/api/v1`

## Route model

Participant routes:

- `/`
- `/consent`
- `/profile`
- `/block-a`
- `/block-a-feedback`
- `/block-b`
- `/block-b-feedback`
- `/complete`

Admin routes:

- `/admin/login`
- `/admin`
