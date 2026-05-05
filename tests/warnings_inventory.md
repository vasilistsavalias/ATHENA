# Warning Inventory

## Purpose

This file tracks warning debt that appears during validation runs and distinguishes first-party issues from third-party/transitive noise.

## Authoritative environment

- Full pipeline: Linux/GCP
- Full repository test suite: repository `.venv`

## Current baseline

Local repository-v-env run on `2026-03-07`:

- command: `.\.venv\Scripts\python.exe -m pytest -q`
- result: `81 passed`
- warnings count: `59`

## First-party warning work completed

### Plotting deprecations fixed

Patched in `src/thesis_pipeline/visualization/plots.py`:

- removed categorical `palette` usage without explicit `hue`
- added explicit `observed=False` to grouped categorical aggregation

These changes address the runtime warnings previously seen during Stage 15 reporting and Stage 13 chart generation.

### Windows Torch collection hardening

Patched tests so torch-dependent modules follow the same subprocess-probe skip strategy:

- `tests/unit/test_stage12_trial_selection.py`
- `tests/unit/test_stage13_all_vs_all_stats.py`
- `tests/unit/test_training_signal_hardening.py`

## Remaining warnings policy

Remaining warnings are treated as lower priority unless they are:

- emitted by ATHENA code directly
- newly introduced by a patch
- promoted to errors by an upstream dependency

## Follow-up rule

If a future validation run shows new first-party warnings from ATHENA code, update this file and either:

- fix them immediately, or
- record why they are intentionally deferred
