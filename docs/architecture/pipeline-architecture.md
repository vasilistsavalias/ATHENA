# Pipeline Architecture

ATHENA's pipeline is artifact-driven and stage-oriented. This document explains the runtime layers, execution path, design decisions behind the architecture, and the contracts that hold the system together.

---

## Why this architecture?

The pipeline had several competing requirements:

1. **Reproducibility** — any researcher must be able to re-run from scratch and get the same result, or resume from any failed stage.
2. **Auditability** — every artefact must be traceable to a stage and a config version.
3. **Flexibility** — stages must be runnable individually for debugging without breaking the full-run mode.
4. **GPU efficiency** — preparation stages (S00–S12) and post-training stages (S14–S18) run on standard Python; only S13 requires distributed Accelerate. The wrapper script routes them separately.

These requirements ruled out monolithic notebook-based pipelines (not resumable, not auditable) and ruled out full orchestration frameworks like Airflow (too heavy for a single-VM research project). The result is a simple directed linear DAG with a run-state ledger.

---

## Runtime layers

```
scripts/pipeline/setup_and_run.sh
  │
  │  environment setup (venv, apt, iopaint, pytest gate)
  │
  └─► run.py
        │
        │  preflight check
        │  git state capture
        │
        └─► src/thesis_pipeline/pipeline/app.py
              │
              │  reads run_state.json (resume mode)
              │  dispatches stages from registry
              │
              └─► src/thesis_pipeline/pipeline/stage_groups.py
                    │
                    ├─► src/thesis_pipeline/stages/s00_preflight.py
                    ├─► src/thesis_pipeline/stages/s01_research_design.py
                    ├─► ...
                    └─► src/thesis_pipeline/stages/s18_expert_pack.py
                              │
                              └─► src/thesis_pipeline/components/*
```

### Layer responsibilities

**`scripts/pipeline/setup_and_run.sh`** — operator entrypoint. Handles everything a fresh Linux VM needs: apt packages, Python venv creation, IOPaint venv setup, pytest gate, GPU selection via `nvidia-smi`, and routing S13 through `accelerate launch`. This is intentionally a shell script rather than Python: it needs to create the Python environment before Python is available.

**`run.py`** — Python bootstrap. Creates/refreshes `.venv`, installs `pipeline_requirements.txt`, runs the preflight script, captures git state, then invokes `main.py` inside the venv. Exists because `setup_and_run.sh` needs to delegate to Python for cross-platform setup logic.

**`src/thesis_pipeline/pipeline/app.py`** — orchestrator. Reads `config/pipeline/main_config.yaml` (or the override), loads run-state from `outputs/00_logs/run_state.json`, resolves which stages to run (accounting for `--resume`, `--phase`, `--force-rerun-successful`), and dispatches them in order.

**`src/thesis_pipeline/pipeline/registry.py`** — stage registry. Maps stage IDs (S00–S18) to their implementing classes. Adding a new stage requires registering it here; the preflight test asserts that the count equals 19.

**`src/thesis_pipeline/stages/`** — stage implementations. Each file implements a single stage as a class with a `run(config, state)` method. Stages write their outputs to deterministic paths under `outputs/` and `data/intermediate/`.

**`src/thesis_pipeline/components/`** — shared building blocks. Reusable code that multiple stages use: data loaders, metric calculators, model wrappers, mask generators. Stages import components; components do not import stages.

---

## Config system

All tuneable parameters live in `config/pipeline/main_config.yaml`. The config is loaded once at pipeline startup by `src/thesis_pipeline/core/config_manager.py` and passed as an immutable object to every stage.

**Why a single YAML config file?**

- A single file makes the entire experiment configuration visible in one place. Comparing two run configurations is a `diff` command, not a search through multiple files.
- The config is validated against a schema at load time (using Pydantic). Invalid configs fail fast with a clear error message, not silently mid-run.

**Config inheritance:**

```
config/pipeline/main_config.yaml      (full production run)
config/pipeline/smoke_test_config.yaml (overrides: 50 images, 2 epochs)
config/pipeline/gcp_resilient.yaml   (overrides: Europeana policy flags)
```

Custom overrides are passed via `--config <path>` and merged on top of the base config.

---

## Run-state ledger

`outputs/00_logs/run_state.json` tracks which stages have completed, their start/end timestamps, and a hash of their primary output artefact. This is the source of truth for `--resume` mode.

```json
{
  "completed_stages": ["S00", "S01", "S02", "S03"],
  "stage_metadata": {
    "S02": {
      "completed_at": "2025-07-15T14:23:11",
      "artifact_hash": "sha256:abc123..."
    }
  }
}
```

**Why a ledger and not checking artifact existence?** Artefact existence is not sufficient — a stage might have written partial output before crashing, leaving files that look complete. The ledger is written only after a stage returns success.

---

## IOPaint separation (`.venv_iopaint`)

S14 uses IOPaint to run LaMa and MAT. IOPaint has dependency conflicts with the main pipeline (different `huggingface_hub` version requirements). Isolating it in `.venv_iopaint` keeps the main venv clean.

The setup script checks IOPaint venv health before every run and rebuilds it if the health check fails. The IOPaint CLI path is exported as `IOPAINT_CLI` for S14 to use.

---

## Distributed training (S13)

S13 is the only stage that uses `accelerate launch`. The wrapper script:

1. Queries `nvidia-smi` for free GPUs (below occupancy threshold)
2. Sets `CUDA_VISIBLE_DEVICES` to the selected GPUs
3. Calls `accelerate.commands.launch --num_processes=N --mixed_precision=fp16`

This design keeps distributed training logic out of the Python pipeline code. The stage itself just calls `accelerate.Accelerator()` and works identically with 1 or N GPUs.

---

## Seed control

`src/thesis_pipeline/utils/seed_manager.py` sets seeds for Python's `random`, `numpy`, `torch`, and CUDA at the start of every stage that produces stochastic output. The seed is derived from `config.seed` (default: 42) combined with the stage index, so different stages use different seeds while remaining reproducible.

---

## Governance preflight assertions

`tests/unit/test_v8_governance_preflight.py` runs a set of assertions that would be easy to accidentally violate:

- Stage count equals 19
- All registered stages have implementations
- Config schema is valid
- Smoke-test fixtures exist and are readable
- Statistical test configurations use the correct test for each metric distribution

These run as part of the S00 pytest gate, ensuring the system is self-consistent before any compute begins.

---

## Key design principles

**Stages are idempotent.** Re-running a stage produces the same outputs. This enables `--resume` and `--force-rerun-successful` to work correctly.

**Stages fail loud.** Stages raise exceptions on unexpected conditions rather than writing partial results and continuing. The orchestrator catches exceptions, updates the run-state ledger (marking the stage failed), and exits.

**No implicit state.** Stages do not read from other stages' in-memory objects. All inter-stage communication happens through files on disk (the artifact paths defined in the config). This makes stages independently testable and resumable.

**No `--clean` in orchestration.** Artifact deletion is performed only by `scripts/pipeline/clean_all_outputs.sh`, which requires an explicit invocation. This prevents accidental deletion of 20 hours of compute when a flag is mistyped.

---

## See also

- [`../../src/thesis_pipeline/pipeline/README.md`](../../src/thesis_pipeline/pipeline/README.md) — orchestrator internals
- [`../../src/thesis_pipeline/stages/README.md`](../../src/thesis_pipeline/stages/README.md) — per-stage contracts
- [`../../src/thesis_pipeline/components/README.md`](../../src/thesis_pipeline/components/README.md) — reusable components
- [`../guides/pipeline-workflow.md`](../guides/pipeline-workflow.md) — narrative stage-by-stage walkthrough
- [`data-models.md`](data-models.md) — persisted state shapes
