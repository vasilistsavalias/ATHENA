# Observability

## Core evidence

- `outputs/00_logs/run_state.json`
- `outputs/00_logs/execution_times.csv`
- `outputs/00_logs/stage_*.log`
- `outputs/00_reproducibility/` or `outputs/S00_reproducibility/`

## Stage-specific evidence

Captioning:

- `outputs/S07_caption_generation/caption_generation_report.json`
- `outputs/S07_caption_generation/caption_quality_report.json`
- `outputs/S08_caption_refinement/caption_refinement_report.json`

Evaluation:

- `outputs/S15_model_evaluation/benchmarking_matrix/caption_coverage_report.json`
- `outputs/S15_model_evaluation/benchmarking_matrix/matrix_results.csv`

## Operating rule

- trust the ledger before memory
- trust persisted artifacts before terminal scrollback
- trust explicit contract checks before visual guesses
