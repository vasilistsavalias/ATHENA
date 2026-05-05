# Components

Reusable building blocks used by stage implementations.

Current subdomains:

- `captioning/` — BLIP2/Qwen local caption pipeline
- `evaluation/` — baselines, metrics, composite ranking, cross-validation
- `filtering/` — filtering audit helpers
- `baseline_finetuning/` — LaMa/MAT/CoModGAN adapter interface

Top-level modules in this folder remain mixed legacy utilities because multiple stages still depend on them directly.
