# ATHENA Web

Next.js 15 frontend for the participant evaluation wizard and admin dashboard.

## Local setup

```bash
cd services/web
cp .env.example .env.local       # set BACKEND_URL=http://localhost:8000
npm install
npm run dev
```

App: http://localhost:3000  
Admin: http://localhost:3000/admin

## Environment variables

| Variable | Description |
|---|---|
| `BACKEND_URL` | Backend base URL for server-side proxy rewrites |
| `NEXT_PUBLIC_API_BASE_URL` | API path prefix shown to the browser (default `/api/v1`) |
| `NEXT_PUBLIC_STUDY_TITLE` | Study title shown in the UI |
| `NEXT_PUBLIC_BLOCK_A_COUNT` | Expected Block A items (display only) |
| `NEXT_PUBLIC_BLOCK_B_COUNT` | Expected Block B items (display only) |

## Participant flow

```
LANDING → CONSENT → PROFILE → BLOCK_A → BLOCK_A_FEEDBACK
       → BLOCK_B (comprehension gate) → BLOCK_B_FEEDBACK
       → BLOCK_C (if campaign has block_c_total > 0)
       → COMPLETE
```

## Testing

```bash
npm run test          # Vitest unit tests
npm run e2e           # Playwright E2E tests (requires running backend)
```

## Docs

- [Study flow](../../docs/study-flow.md)
- [Architecture](../../docs/architecture.md)
- [API contracts](../../docs/api-contracts.md)
