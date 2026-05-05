# ATHENA: Reproducible Inpainting Benchmark & Expert Evaluation Platform

An end-to-end framework to benchmark AI inpainting on cultural heritage artifacts. This was built for a master's thesis at the University of Macedonia (AID25006).

The project has two main parts. Use one or both:

1. **ML Pipeline**: A 19-stage pipeline covering data acquisition, filtering, training, and evaluation.
2. **Evaluation Platform**: A web app for expert studies with an admin dashboard and export tools.

---

```text
  Wikimedia Commons / Europeana API
           │
  ┌────────▼────────────────────────────────────────────────────────────┐
  │                        ML PIPELINE  (Linux / GCP)                   │
  │                                                                      │
  │  S00 Preflight ──► S01 Research Design                               │
  │       │                                                              │
  │  S02 Data Acquisition (Wikimedia + Europeana)                        │
  │       │                                                              │
  │  S03 Intelligent Filtering (YOLOv8x + bias audit)                    │
  │       │                                                              │
  │  S04 YOLO Validation ──► S05 Classical Baselines                     │
  │       │                                                              │
  │  S06 Exploratory Data Analysis                                       │
  │       │                                                              │
  │  S07 Caption Generation (BLIP-2) ──► S08 Refinement (Qwen2.5-7B)    │
  │       │                                                              │
  │  S09 Data Processing (512×512 PNG) ──► S10 Splitting (80/10/10)     │
  │       │                                                              │
  │  S11 Mask Generation (irregular / rect / edge)                       │
  │       │                                                              │
  │  S12 Hyperparameter Tuning (8-trial sweep)                           │
  │       │                                                              │
  │  S13 Model Training (distributed, Accelerate, K-fold)               │
  │       │                                                              │
  │  S14 Baseline Fine-tuning (LaMa · MAT · CoModGAN)                   │
  │       │                                                              │
  │  S15 Full Evaluation Matrix + Deep Baselines                         │
  │       │                                                              │
  │  S16 Deployment Artefacts (ONNX / PT export)                        │
  │       │                                                              │
  │  S17 Reporting (plots, samples, artefacts)                           │
  │       │                                                              │
  │  S18 Expert Pack generation (.zip)  ◄── feeds platform below        │
  └───────────────────────────┬──────────────────────────────────────────┘
                              │
              ┌───────────────▼──────────────────────┐
              │     EVALUATION PLATFORM  (web)        │
              │                                       │
              │  Admin imports pack                   │
              │          │                            │
              │  Experts invited via invite code      │
              │          │                            │
              │   ┌──────▼──────────────────────┐     │
              │   │  Block A: Authenticity &     │     │
              │   │  plausibility rating         │     │
              │   │  (Likert + required comment) │     │
              │   └──────────────┬───────────────┘     │
              │                  │                      │
              │   ┌──────────────▼───────────────┐     │
              │   │  Block B: Pairwise           │     │
              │   │  preference comparison        │     │
              │   │  (comprehension gate first)  │     │
              │   └──────────────┬───────────────┘     │
              │                  │                      │
              │   ┌──────────────▼───────────────┐     │
              │   │  Block C: Four-way model     │     │
              │   │  selection                   │     │
              │   └──────────────┬───────────────┘     │
              │                  │                      │
              │  Admin exports CSV / JSON / QR          │
              └─────────────────────────────────────────┘
```

---

## ML Pipeline: Quickstart

Use this entry point to run the whole pipeline:

```bash
# Full run (Linux / GCP)
chmod +x scripts/pipeline/setup_and_run.sh
./scripts/pipeline/setup_and_run.sh --full

# Smoke test: Fast validation using a 50-image subset
./scripts/pipeline/setup_and_run.sh --smoke-test

# Resume a partial run from a specific stage
./scripts/pipeline/setup_and_run.sh --resume --phase S13

# Windows (PowerShell)
.\scripts\pipeline\setup_and_run.ps1
```

The script sets up the venv, installs dependencies, and runs preflight checks automatically. No manual `pip install` required.

### Configuration

| File | Purpose |
|---|---|
| `config/pipeline/main_config.yaml` | Production settings |
| `config/pipeline/smoke_test_config.yaml` | Quick 50-image dev/validation run |

### Reproducibility

- Controlled by seeds in `src/thesis_pipeline/utils/seed_manager.py`.
- Includes a state ledger to resume from any stage using `--resume`.
- Preflight pytest gate runs before any compute stage.

### GCP deployment

Check [`docs/gcp-pipeline.md`](docs/gcp-pipeline.md) for the GCP setup guide (SSH, cloning, GPU selection, and Europeana-resilient mode).

### Pipeline internals

| Doc | Contents |
| --- | --- |
| [`docs/getting-started/quickstart.md`](docs/getting-started/quickstart.md) | Command reference |
| [`docs/getting-started/codebase-tour.md`](docs/getting-started/codebase-tour.md) | Layout, tech stack, and rationale |
| [`docs/guides/pipeline-workflow.md`](docs/guides/pipeline-workflow.md) | Stage explanations and design choices |
| [`docs/architecture/pipeline-architecture.md`](docs/architecture/pipeline-architecture.md) | Runtime layers and config system |
| [`docs/gcp-pipeline.md`](docs/gcp-pipeline.md) | VM setup and results retrieval |
| [`docs/reference/commands.md`](docs/reference/commands.md) | Flags and CLI arguments |
| [`docs/reference/artifacts.md`](docs/reference/artifacts.md) | Output directory reference |
| [`docs/reference/troubleshooting.md`](docs/reference/troubleshooting.md) | Diagnostics |
| [`docs/reference/glossary.md`](docs/reference/glossary.md) | Terms and acronyms |
| [`src/thesis_pipeline/pipeline/README.md`](src/thesis_pipeline/pipeline/README.md) | Orchestration details |
| [`src/thesis_pipeline/stages/README.md`](src/thesis_pipeline/stages/README.md) | Stage contracts and artifacts |
| [`config/pipeline/README.md`](config/pipeline/README.md) | Config schema documentation |

---

## Evaluation Platform: Quickstart

### Local (no Docker)

**Backend:**

```bash
cd services/api
python -m venv .venv
source .venv/bin/activate        # Windows: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
cp .env.example .env             # configure secrets
alembic upgrade head
uvicorn app.main:app --reload --port 8000
```

**Frontend:**

```bash
cd services/web
cp .env.example .env.local       # set BACKEND_URL=http://localhost:8000
npm install
npm run dev
```

App: <http://localhost:3000> · Admin: <http://localhost:3000/admin> · API docs: <http://localhost:8000/docs>

### Docker (web platform only)

```bash
cp .env.example .env             # configure secrets
docker compose up --build
```

### Key environment variables

| Variable | Where | Description |
|---|---|---|
| `DATABASE_URL` | backend | PostgreSQL DSN (SQLite by default) |
| `APP_INVITE_CODE` | backend | Registration code for participants |
| `APP_SHARED_PASSWORD` | backend | Shared login password |
| `SESSION_SECRET` | backend | Key for signing sessions |
| `ADMIN_UI_PASSWORD` | backend | Admin dashboard password |
| `ALLOWED_ORIGINS` | backend | CORS allowed origins |
| `BACKEND_URL` | frontend | Backend URL for the proxy |

Full reference in `.env.example`, `services/api/.env.example`, and `services/web/.env.example`.

### Loading a campaign

ATHENA uses **campaign packs**: zip archives with a manifest and images generated in S18.

1. **Admin UI**: Go to `/admin` and use "Import Pack".
2. **Bootstrap on startup**: Set `BOOTSTRAP_PACK_ON_STARTUP=true` and `BOOTSTRAP_PACK_ZIP_PATH=<path>`.

Pack format details: [`services/api/bootstrap/README.md`](services/api/bootstrap/README.md)

### Platform internals

| Doc | Contents |
| --- | --- |
| [`services/api/README.md`](services/api/README.md) | Backend setup and protocols |
| [`services/web/README.md`](services/web/README.md) | Frontend setup and participant flow |
| [`docs/study-flow.md`](docs/study-flow.md) | Wizard flow and data collection |
| [`docs/guides/admin-operations.md`](docs/guides/admin-operations.md) | Admin tasks and privacy |
| [`docs/api-contracts.md`](docs/api-contracts.md) | API endpoint reference |
| [`docs/architecture.md`](docs/architecture.md) | Service layout and routing |
| [`docs/deployment.md`](docs/deployment.md) | Production checklist |
| [`docs/reference/security.md`](docs/reference/security.md) | Security and data boundaries |

---

## Repository layout

```text
ATHENA/
├── src/thesis_pipeline/       ← Pipeline Python package (S00–S18)
│   ├── stages/                ← Stage-specific logic
│   ├── components/            ← Reusable ML modules
│   ├── pipeline/              ← Orchestrator and registry
│   ├── analysis/              ← Statistical analysis
│   ├── utils/                 ← Utilities (seeds, tracking)
│   └── visualization/         ← Plotting code
├── scripts/
│   ├── pipeline/              ← Entry points and shell helpers
│   ├── utilities/             ← Pack builders and export tools
│   └── repo/                  ← Audit and maintenance scripts
├── config/pipeline/           ← YAML configurations
├── tests/                     ← Test suite (121 tests)
├── public_fixtures/           ← Small fixtures for testing
├── services/
│   ├── api/                   ← FastAPI backend
│   └── web/                   ← Next.js 15 frontend
├── docs/                      ← Project documentation
├── run.py                     ← Python entry point
├── pipeline_requirements.txt  ← Pipeline dependencies
├── docker-compose.yml         ← Docker setup for the web platform
└── .env.example               ← Environment template
```

---

## Tests

```bash
# Requires pipeline dependencies
pytest tests/ -q
```

Includes 121 tests for stages, statistical rigor, and governance.

---

## License

GNU General Public License v3.0: see [LICENSE](LICENSE).

Copyright © 2025 Vasilis Tsavalias
