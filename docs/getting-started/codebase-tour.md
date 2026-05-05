# Codebase Tour

ATHENA is two independent systems in one repository: a 19-stage ML pipeline and a web-based expert evaluation platform. You can use either or both. This tour explains where everything lives, what each part does, and why it was structured this way.

---

## Repository layout

```
ATHENA/
├── src/thesis_pipeline/       ML pipeline Python package
│   ├── stages/                One file per stage (S00–S18)
│   ├── components/            Reusable ML/data/evaluation building blocks
│   ├── pipeline/              Orchestrator, stage registry, governance
│   ├── analysis/              Post-hoc statistical analysis utilities
│   ├── utils/                 Seed manager, resource tracker, helpers
│   └── visualization/         Matplotlib plotting helpers
│
├── scripts/
│   ├── pipeline/              Operator entrypoints (setup_and_run.sh/.ps1)
│   ├── utilities/             Pack builders, export tools, admin helpers
│   └── repo/                  Weight download, audit scripts
│
├── config/pipeline/           YAML configs (main + smoke_test)
│
├── tests/                     pytest suite (121 tests)
├── public_fixtures/           2-image smoke-test dataset
│
├── services/
│   ├── api/                   FastAPI backend (evaluation platform)
│   └── web/                   Next.js 15 frontend (evaluation platform)
│
├── docs/                      This documentation
│   ├── getting-started/       Quickstart and this tour
│   ├── guides/                Pipeline workflow, study flow, admin ops
│   ├── architecture/          Architecture docs with design decisions
│   └── reference/             Commands, artifacts, glossary, troubleshooting
│
├── run.py                     Python bootstrap entry point for pipeline
├── pipeline_requirements.txt  Pipeline Python dependencies
├── docker-compose.yml         Web platform only (Postgres + API + frontend)
└── .env.example               Environment variable template
```

---

## The ML pipeline

### Entry points

**`scripts/pipeline/setup_and_run.sh`** — the primary operator command on Linux/GCP. Handles venv creation, dependency installation, pytest gate, GPU selection, and stage dispatch. Start here for any GCP run.

**`scripts/pipeline/setup_and_run.ps1`** — Windows PowerShell equivalent. Simpler than the `.sh` version: sets up venv and runs the pipeline. Windows is primarily used for development, not production GPU runs.

**`run.py`** — Python-level bootstrap. Creates `.venv`, installs `pipeline_requirements.txt`, runs preflight, captures git state, then invokes the pipeline. Called internally by `setup_and_run.sh`.

### Pipeline package: `src/thesis_pipeline/`

**`pipeline/`** — the orchestrator. `app.py` is the main entry point for the pipeline itself; `registry.py` maps stage IDs to implementations; `stage_groups.py` handles group dispatch (preparation / training / post-training).

**`stages/`** — one `.py` file per stage (`s00_preflight.py` through `s18_expert_pack.py`). Each stage class has a `run(config, state)` method. Stages are the only place where pipeline logic lives; they call into `components/` for reusable work.

**`components/`** — shared building blocks used by multiple stages. Examples:
- `data_acquisition.py` — Wikimedia and Europeana API clients
- `mask_generator.py` — irregular, rectangular, and edge mask generation
- `evaluation_matrix.py` — PSNR, SSIM, LPIPS, FID, KID computation
- `diffusion_trainer.py` — LoRA fine-tuning wrapper for Stable Diffusion

**`core/`** — infrastructure. `config_manager.py` loads and validates YAML configs; `logging_config.py` sets up structured logging with per-stage log files.

**`utils/`** — helpers. `seed_manager.py` is the most important: it sets deterministic seeds for every random library (Python, NumPy, PyTorch, CUDA) at the start of each stage.

**`analysis/`** — post-hoc statistical analysis. Wilcoxon tests, Bonferroni correction, effect size calculations. Imported by S15 evaluation.

**`visualization/`** — Matplotlib plotting helpers. Used by S17 reporting to generate every figure in the thesis.

### Configuration

**`config/pipeline/main_config.yaml`** — full production configuration. Every tuneable parameter for every stage lives here. Read once at startup, validated against a Pydantic schema, passed immutably to all stages.

**`config/pipeline/smoke_test_config.yaml`** — overrides for fast validation: 50 images, 2 epochs, 2 K-fold splits.

### Tests

**`tests/`** — 121 tests covering every stage, statistical rigor, governance assertions, and unit contracts. Run as a mandatory gate before any pipeline execution via `setup_and_run.sh`.

- `tests/unit/test_v8_governance_preflight.py` — asserts stage count = 19, all stages registered, config schema valid
- `tests/unit/test_statistical_rigor.py` — asserts correct statistical tests are used for each metric
- `tests/unit/test_stage13_*.py` — distributed training contract tests

**`public_fixtures/`** — 2 images and minimal metadata used by smoke-test fixtures. Tiny enough to commit to the repo; just large enough to exercise the full pipeline.

---

## The evaluation platform

### Entry points

**Backend:** `services/api/` — FastAPI application. Start with `uvicorn app.main:app --reload --port 8000`.

**Frontend:** `services/web/` — Next.js 15. Start with `npm run dev` (runs on port 3000).

**Docker:** `docker-compose.yml` starts PostgreSQL + FastAPI backend + Next.js frontend. Does not run the ML pipeline.

### Backend: `services/api/`

**`app/main.py`** — FastAPI application factory. Registers routers, runs startup hooks (bootstrap pack import if configured).

**`app/api/endpoints/`** — all HTTP endpoints:
- `sessions.py` — participant registration, session state
- `assignments.py` — item assignment per block per session
- `submissions.py` — Block A/B/C response submission
- `admin.py` — admin dashboard data, pack import, export
- `exports.py` — CSV/JSON export endpoints

**`app/models/`** — SQLAlchemy ORM models (Campaign, Session, Assignment, Submission, etc.)

**`app/services/`** — business logic layer between endpoints and ORM.

**`app/core/config.py`** — Pydantic settings loaded from environment variables.

**`bootstrap/`** — startup pack import. If `BOOTSTRAP_PACK_ON_STARTUP=true`, imports a campaign zip on first start. Used for ephemeral Render hosting where the SQLite database resets on every deploy.

**`alembic/`** — database migrations.

### Frontend: `services/web/`

**`src/app/`** — Next.js App Router pages.

**`src/components/`** — shared UI components.

**`src/lib/`** — API client, state management, utilities.

Key pages:
- `/` — landing page (invite code entry)
- `/consent` — informed consent
- `/profile` — participant background form
- `/block-a`, `/block-b`, `/block-c` — study blocks
- `/complete` — completion confirmation
- `/admin` — admin dashboard (password-protected)

**Why Next.js 15?** Server-side rendering is important for the evaluation platform because participants may have slow connections. Next.js gives us SSR, API route proxying (the frontend proxies all API calls to avoid CORS complexity), and React Server Components for fast initial page loads.

**Why a separate frontend?** The admin dashboard and participant flow have very different rendering requirements. A pure React SPA would work but loses SSR benefits on the participant side. Keeping them as Next.js pages in the same app lets us share components and session state while getting SSR where it matters.

---

## Why two systems in one repo?

The ML pipeline (S18) produces the campaign pack that the evaluation platform imports. Keeping them in the same repository:

- Makes it easier to verify the pack format is in sync with the importer
- Allows shared documentation and a single release
- Avoids the version-pinning complexity of two separate repos with a wire format dependency

They are genuinely independent runtime systems (no shared code, different languages, different deployment targets), so they can also be extracted into separate repos easily — the `services/` directory has no imports from `src/thesis_pipeline/`.

---

## Key technologies and why they were chosen

| Component | Technology | Reason |
| --- | --- | --- |
| Inpainting model | Stable Diffusion 2.0 (diffusers) | Best open-weight inpainting model; supports text conditioning; LoRA fine-tuning |
| Object detection | YOLOv8x | Provides bounding boxes for spatial mask placement; best accuracy at the xlarge scale |
| Caption generation | BLIP-2 (2.7B) | Open-weight, runs locally, produces fluent descriptions; fits on single A100 |
| Caption refinement | Qwen2.5-7B | Strong instruction-following for structured reformatting; fits on single A100 in fp16 |
| Distributed training | HuggingFace Accelerate | Abstracts NCCL + mixed precision into `accelerate launch`; minimal boilerplate |
| Hyperparameter search | Optuna (TPE sampler) | Bayesian optimisation extracts more signal per trial than grid/random search |
| Backend framework | FastAPI | Automatic OpenAPI docs; async support; Pydantic validation |
| Database (dev) | SQLite | Zero-configuration; no separate service needed for local development |
| Database (prod) | PostgreSQL | Required for concurrent write performance with multiple participants |
| Frontend framework | Next.js 15 | SSR for participant pages; API proxying; React Server Components |
| Statistical tests | Wilcoxon signed-rank | Non-parametric; correct for bounded metric distributions (PSNR, SSIM, LPIPS) |

---

## Where to go next

- **Running the pipeline:** [`getting-started/quickstart.md`](quickstart.md)
- **Understanding each stage:** [`guides/pipeline-workflow.md`](../guides/pipeline-workflow.md)
- **Pipeline architecture deep-dive:** [`architecture/pipeline-architecture.md`](../architecture/pipeline-architecture.md)
- **Web platform participant flow:** [`guides/study-flow.md`](../../docs/study-flow.md)
- **GCP deployment:** [`gcp-pipeline.md`](../gcp-pipeline.md)
- **All flags and commands:** [`reference/commands.md`](../reference/commands.md)
