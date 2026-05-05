# ATHENA API

FastAPI backend for participant sessions, assignments, stage feedback, admin analytics, campaign import, and exports.

## Local setup

```bash
cd services/api
python -m venv .venv
source .venv/bin/activate        # Windows: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
cp .env.example .env             # edit secrets
alembic upgrade head
uvicorn app.main:app --reload --port 8000
```

API docs: http://localhost:8000/docs

## Bootstrap (ephemeral hosting)

Set `BOOTSTRAP_PACK_ON_STARTUP=true` and place a pack zip at `bootstrap/final_expert_pack.zip` (or set `BOOTSTRAP_PACK_ZIP_PATH`). The backend auto-imports at startup when no active campaign exists or assets are missing.

See [`bootstrap/README.md`](bootstrap/README.md) for pack format details.

## Key protocol behaviour

- Block A and B submissions require a non-empty trimmed comment
- Block B scored items require comprehension completion before the first scored pair
- Three-part study mode supported: Block B (pairwise) + Block C (four-way model selection)
- Exports include `block_b_comprehension_attempts`, `comprehension_risk`, and exceedance report

## Docs

- [API contracts](../../docs/api-contracts.md)
- [Architecture](../../docs/architecture.md)
- [Admin operations](../../docs/admin-operations.md)
- [Deployment](../../docs/deployment.md)
