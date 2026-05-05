# Core

This package contains shared runtime primitives used by the entire pipeline.

- `config.py` — canonical YAML loading and section access
- `logging.py` — canonical logging setup

Backward-compatible imports remain available through:

- `thesis_pipeline.config_manager`
- `thesis_pipeline.logging_config`
