# Pipeline Configuration Guide

This directory is the runtime control center for `thesis_pipeline`.

## Main files

- `main_config.yaml` — production/default configuration
- `smoke_test_config.yaml` — lightweight test profile
- `src/thesis_pipeline/core/logging.py` — logging setup consumed by the runtime

## Important config domains

- `pipeline.*` — strict/resume behavior and runtime policy
- `data_acquisition.*` — Wikimedia/Europeana controls, limits, query variants
- `hyperparameter_tuning.*` — `S05` sweep strategy and search space
- `model_training.*` — `S06` checkpoint/resume and trial policy
- `model_evaluation.*` — `S08` phase, baseline requirements, significance controls
- `paths.*` — canonical outputs and intermediate-data directories

## Override patterns

Selected runtime behavior can be overridden from the CLI:

- `--phase integrity_200|full_test`
- `--stage02-limit N`
- `--raw-dir <path>`
- `--artifacts-root <path>`

Always prefer config plus explicit CLI overrides over ad-hoc code changes.
