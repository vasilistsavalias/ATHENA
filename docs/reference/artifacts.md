# Artifacts Reference

## Canonical artifact roots

- raw acquisitions: `data/01_raw/combined_collection`
- intermediates: `data/intermediate/`
- stage outputs: `outputs/S00_*` to `outputs/S18_*`
- website local runtime data: `website/services/api/data/`

## Key contracts

Filtering:

- `data/intermediate/02_filtered/accepted`
- `data/intermediate/02_filtered/rejected`

Captioning and refinement:

- `outputs/S07_caption_generation/captions_raw.json`
- `outputs/S07_caption_generation/captions_enriched.json`
- `outputs/S07_caption_generation/caption_generation_report.json`
- `outputs/S08_caption_refinement/caption_refinement_report.json`

Splits and masks:

- `data/intermediate/07_splits/{train,validation,test}`
- `data/intermediate/08_inpainting/{train,validation,test}/`

Training and tuning:

- `outputs/S12_hyperparameter_tuning/sweep_plan.yaml`
- `outputs/S12_hyperparameter_tuning/best_hyperparameters.yaml`
- `data/intermediate/10_models/unet_best`
- `data/intermediate/10_models/unet_final`

Evaluation:

- `outputs/S15_model_evaluation/benchmarking_matrix/matrix_results.csv`
- `outputs/S15_model_evaluation/benchmarking_matrix/caption_coverage_report.json`
- `outputs/S15_model_evaluation/statistical_tests/*.csv`

Expert pack:

- `outputs/S18_expert_validation/*`

Website exports:

- `responses.csv`
- `responses.json`
- `quality_report.json`

Canonical sources:

- `config/pipeline/main_config.yaml`
- `src/thesis_pipeline/pipeline/registry.py`
