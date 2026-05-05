from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock


_LEGACY_STAGE_KEYS: dict[str, list[str]] = {
    "S07": ["stage_07", "stage_06"],
    "S08": ["stage_08", "stage_07", "stage_06"],
    "S11": ["stage_11", "stage_10"],
    "S12": ["stage_12", "stage_11"],
    "S15": ["stage_15", "stage_13", "stage_11"],
}


def _read_mapping_value(container: Any, key: str) -> Any:
    if container is None:
        return None
    if isinstance(container, dict):
        return container.get(key)
    if isinstance(container, SimpleNamespace):
        return getattr(container, key, None)
    return getattr(container, key, None)


def _coerce_path(value: Any) -> Path | None:
    if isinstance(value, Mock):
        return None
    if isinstance(value, Path):
        return value
    if isinstance(value, str):
        return Path(value)
    if isinstance(value, os.PathLike):
        return Path(value)
    return None


def resolve_stage_artifact_dir(config_manager: Any, stage_id: str) -> Path:
    getter = getattr(config_manager, "get_stage_artifact_dir", None)
    if callable(getter):
        try:
            resolved = _coerce_path(getter(stage_id))
            if resolved is not None:
                return resolved
        except Exception:
            pass

    paths_getter = getattr(config_manager, "get_paths", None)
    paths = paths_getter() if callable(paths_getter) else getattr(config_manager, "paths", None)
    artifacts = _read_mapping_value(paths, "artifacts")

    for key in _LEGACY_STAGE_KEYS.get(stage_id, []):
        resolved = _coerce_path(_read_mapping_value(artifacts, key))
        if resolved is not None:
            return resolved

    root = _coerce_path(_read_mapping_value(artifacts, "root"))
    if root is not None:
        return root

    return Path("outputs") / stage_id
