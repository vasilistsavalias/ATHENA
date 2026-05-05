import os
from pathlib import Path


def load_env_file(dotenv_path: Path, *, override: bool = False) -> list[str]:
    """Best-effort `.env` loader (no external deps).

    Security notes
    --------------
    - This function never logs secret values.
    - It is intended for local/dev convenience only; CI/GCP should use real env vars.

    Parsing rules
    -------------
    - Lines starting with `#` are ignored.
    - `KEY=VALUE` pairs are loaded.
    - If a line contains no `=`, and `EUROPEANA_API_KEY` is not set, treat the
      whole line as the Europeana key (supports the user's "value-only" `.env`).
    """
    loaded_keys: list[str] = []
    try:
        if not dotenv_path or not Path(dotenv_path).exists():
            return loaded_keys

        for raw_line in Path(dotenv_path).read_text(encoding="utf-8", errors="ignore").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue

            if "=" in line:
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if not key:
                    continue
                if override or key not in os.environ:
                    os.environ[key] = value
                    loaded_keys.append(key)
                continue

            # Value-only fallback: treat as Europeana API key.
            if override or "EUROPEANA_API_KEY" not in os.environ:
                os.environ["EUROPEANA_API_KEY"] = line.strip().strip('"').strip("'")
                loaded_keys.append("EUROPEANA_API_KEY")
    except Exception:
        # Intentionally swallow errors: `.env` must never break the pipeline.
        return loaded_keys

    return loaded_keys

