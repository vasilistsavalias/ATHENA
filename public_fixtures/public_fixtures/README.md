# Public Fixtures

This folder contains the only in-repo sample assets intended for quick public validation.

## Contents

- `images/smoke_seed_00.jpg`
- `images/smoke_seed_01.jpg`
- `metadata/smoke_seed_00.json`
- `metadata/smoke_seed_01.json`

These fixtures are intentionally tiny and non-representative. They are for smoke checks only.

## Weights policy

Model weights are not tracked in git history. Use the helper script:

- PowerShell: `scripts/repo/download_public_weights.ps1`

## Reproducibility note

If you need full experiments, run the pipeline and generate local data/outputs under ignored paths.
