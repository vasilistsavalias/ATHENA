"""Canonical pipeline orchestration namespace."""

from thesis_pipeline.pipeline.registry import (
    LEGACY_TO_NEW,
    STAGE_BY_ID,
    STAGE_CATALOG,
    STAGE_ORDER,
    StageID,
    StageInfo,
    execution_order,
    output_dir_for,
    resolve_stage_id,
)


def main(*args, **kwargs):
    from thesis_pipeline.pipeline.app import main as _main

    return _main(*args, **kwargs)


def cli(*args, **kwargs):
    from thesis_pipeline.pipeline.app import cli as _cli

    return _cli(*args, **kwargs)

__all__ = [
    "cli",
    "main",
    "StageID",
    "StageInfo",
    "STAGE_CATALOG",
    "STAGE_ORDER",
    "STAGE_BY_ID",
    "LEGACY_TO_NEW",
    "execution_order",
    "output_dir_for",
    "resolve_stage_id",
]
