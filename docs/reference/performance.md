# Performance

## Main cost drivers

- `S01` acquisition — network latency and source throttling
- `S03` captioning/refinement — BLIP2 VRAM pressure, Qwen generation cost
- `S06` training — FT-SD epochs, validation cadence, checkpoints
- `S08` evaluation — model-family breadth and all-vs-all expansion

## Main tuning knobs

- `caption_generation.max_new_tokens`
- `caption_generation.oom_retry_limit`
- `training.train_batch_size`
- `training.num_epochs`
- `model_evaluation.num_samples_to_evaluate`
- `baseline_finetuning.batch_size`

## Practical rule

Successful stages should be resumed, not rerun.
