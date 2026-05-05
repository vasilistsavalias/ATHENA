# Pipeline

This package is the canonical orchestration surface.

- `app.py` — CLI/orchestrator and run-state ledger
- `registry.py` — canonical `S00`–`S18` stage registry
- `stage_groups.py` — canonical stage bindings to numbered implementation modules

Backward-compatible imports remain available through:

- `thesis_pipeline.main`
- `thesis_pipeline.stage_registry`
- `thesis_pipeline.stages`
