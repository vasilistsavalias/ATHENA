# Deployment

## Backend (Render)

Backend required env vars:

- `DATABASE_URL` — PostgreSQL connection string
- `APP_ENV=production`
- `APP_INVITE_CODE` — Code participants enter to register
- `SESSION_SECRET` — Session signing key (generate a long random string)
- `ADMIN_EXPORT_SECRET` — Admin API secret / fallback UI password
- `ADMIN_UI_PASSWORD` — Admin dashboard password (overrides export secret if set)
- `ALLOWED_ORIGINS=https://<your-app>.vercel.app`
- `SESSION_COOKIE_SAMESITE=none`
- `SESSION_COOKIE_SECURE=true`
- `BOOTSTRAP_PACK_ON_STARTUP=true` (recommended on Render free tier — disk is ephemeral)

Backend optional bootstrap env vars:

- `BOOTSTRAP_PACK_ZIP_PATH` (default: `services/api/bootstrap/final_expert_pack.zip`)
- `BOOTSTRAP_CAMPAIGN_NAME`
- `BOOTSTRAP_CAMPAIGN_SEED`
- `BOOTSTRAP_ACTIVATE=true`
- `BOOTSTRAP_DISJOINT_BLOCKS=true`
- `EXPERT_PROTOCOL_VERSION=ATHENA Expert Protocol v1.1`

## Frontend (Vercel)

Frontend required env vars:

- `BACKEND_URL=https://<your-backend>.onrender.com`
- `NEXT_PUBLIC_API_BASE_URL=/api/v1`
- `NEXT_PUBLIC_STUDY_TITLE=ATHENA`
- `NEXT_PUBLIC_BLOCK_A_COUNT=0`
- `NEXT_PUBLIC_BLOCK_B_COUNT=20`

## Important runtime rule

Browser traffic must reach the backend through the frontend's `/api/v1` proxy (`next.config.js` rewrites). Do not expose the backend URL directly in client-side environment variables.

## Protocol checklist (before inviting participants)

The deployed instance must expose:

- protocol version text during onboarding
- the Block B comprehension gate before the first scored comparison
- practice items before scored pairwise items

Do not recruit additional participants after changing protocol wording unless the protocol version has been bumped.
