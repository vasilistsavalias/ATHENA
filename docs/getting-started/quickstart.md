# Quickstart

Get ATHENA running without reading the full documentation first. For deeper explanation of every choice, see the [Codebase Tour](codebase-tour.md).

---

## Pipeline (Linux / GCP)

This is the recommended path for production runs. See [`docs/gcp-pipeline.md`](../gcp-pipeline.md) for the full GCP setup guide including VM provisioning, tmux workflow, and results retrieval.

```bash
# Clone the repo
git clone https://github.com/YOUR_ORG/athena.git
cd athena

# Make scripts executable
chmod +x scripts/pipeline/setup_and_run.sh
chmod +x scripts/pipeline/clean_all_outputs.sh

# Full run — handles venv, deps, pytest gate, GPU selection automatically
./scripts/pipeline/setup_and_run.sh --full

# With Europeana API key (optional, improves dataset diversity)
EUROPEANA_API_KEY='YOUR_KEY' ./scripts/pipeline/setup_and_run.sh --full
```

**Smoke test** — validates the full pipeline end-to-end with 50 images in ~30 minutes:

```bash
./scripts/pipeline/setup_and_run.sh --smoke-test
```

**Resume** after a crash or interruption:

```bash
./scripts/pipeline/setup_and_run.sh --resume --full
```

**Fresh start** (wipes all outputs):

```bash
./scripts/pipeline/clean_all_outputs.sh
./scripts/pipeline/setup_and_run.sh --full
```

---

## Pipeline (Windows — development only)

Windows does not support the distributed GPU training used in S13. Use this for development, testing, and smoke tests only.

```powershell
# Run tests
.\.venv\Scripts\python.exe -m pytest tests/ -q

# Smoke test
.\scripts\pipeline\setup_and_run.ps1 --smoke-test
```

---

## Evaluation platform — local (no Docker)

**Backend:**

```bash
cd services/api
python -m venv .venv

# Linux/macOS
source .venv/bin/activate
# Windows
.\.venv\Scripts\Activate.ps1

pip install -r requirements.txt
cp .env.example .env   # edit secrets
alembic upgrade head
uvicorn app.main:app --reload --port 8000
```

**Frontend:**

```bash
cd services/web
cp .env.example .env.local   # set BACKEND_URL=http://localhost:8000
npm install
npm run dev
```

App: <http://localhost:3000>  
Admin: <http://localhost:3000/admin>  
API docs: <http://localhost:8000/docs>

---

## Evaluation platform — Docker

Web platform only. Does not run the ML pipeline.

```bash
cp .env.example .env   # edit secrets
docker compose up --build
```

---

## Loading a campaign pack

After S18 runs, import the expert pack into the platform:

**Via admin UI:** navigate to `/admin` → **Import Pack** → upload `outputs/S18_expert_validation/campaign_pack.zip`

**Via bootstrap (Render / ephemeral hosting):**

```bash
# In services/api/.env
BOOTSTRAP_PACK_ON_STARTUP=true
BOOTSTRAP_PACK_ZIP_PATH=/app/bootstrap/final_expert_pack.zip
```

---

## What to read next

| Goal | Read |
| --- | --- |
| Understand every stage and why | [`guides/pipeline-workflow.md`](../guides/pipeline-workflow.md) |
| Full codebase map with rationale | [`getting-started/codebase-tour.md`](codebase-tour.md) |
| GCP VM setup + tmux + results retrieval | [`gcp-pipeline.md`](../gcp-pipeline.md) |
| Architecture and design decisions | [`architecture/pipeline-architecture.md`](../architecture/pipeline-architecture.md) |
| Participant study flow | [`study-flow.md`](../../docs/study-flow.md) |
| All flags and commands | [`reference/commands.md`](../reference/commands.md) |
| Artifact paths reference | [`reference/artifacts.md`](../reference/artifacts.md) |
| Troubleshooting | [`reference/troubleshooting.md`](../reference/troubleshooting.md) |
