"""Shared pytest fixtures for the thesis_pipeline test suite."""
import os
import subprocess
import sys
import pytest


def _torch_import_ok_via_subprocess() -> bool:
    try:
        completed = subprocess.run(
            [sys.executable, "-c", "import torch; print(torch.__version__)"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=30,
        )
        return completed.returncode == 0
    except Exception:
        return False


# Torch import on Windows can hard-crash the interpreter (access violation) if the install is broken.
# We probe in a subprocess first, then let torch-dependent test modules decide whether to skip.
if os.name == "nt":
    exe = sys.executable.lower()
    # Guardrail: many users have a broken torch installed in user site-packages. Running via the repo venv is required.
    if ".venv" not in exe and os.environ.get("FORCE_TORCH_TESTS") != "1":
        os.environ.setdefault("TORCH_IMPORT_OK", "0")
    else:
        os.environ.setdefault("TORCH_IMPORT_OK", "1" if _torch_import_ok_via_subprocess() else "0")


@pytest.fixture(autouse=True, scope="session")
def set_deterministic_seed():
    """Set seeds once for the entire test session.

    Uses SeedManager to mirror the exact seeding the pipeline does at runtime.
    """
    # Torch import on Windows can hard-crash the interpreter (access violation) if the install is broken.
    # Avoid importing torch/SeedManager on Windows; Linux/GCP is the primary execution environment.
    if os.name == "nt":
        return

    # Late import to avoid torch-DLL issues on Windows CI when torch is broken.
    try:
        from thesis_pipeline.utils.seed_manager import SeedManager
        SeedManager.set_seed(42)
    except Exception:
        # Graceful fallback — tests that need seeding will set it themselves.
        pass


@pytest.fixture()
def tmp_dir(tmp_path):
    """Convenience alias for pytest's tmp_path."""
    return tmp_path
