# Analysis Module

This package contains executable post-hoc analysis code only.

## Scope

Use this module after evaluation when you need deeper diagnostics beyond the standard stage outputs.

## Main tools

- `bias_analyzer.py` — category-level disparity checks
- `failure_analyzer.py` — failure case clustering and root-cause slicing
- `statistical_rigor.py` / `statistical_tests.py` — inferential testing helpers
- `domain_metrics.py` — archaeology-oriented quality metrics
- `caption_analysis.py` — caption quality and conditioning diagnostics
- `realism_validator.py` — synthetic-vs-real damage realism checks

## Notes

- This package is optional; the main pipeline can run without it.
- Non-code notes and TODO documents were moved to `notes/analysis/` so that `src/` stays code-first.
