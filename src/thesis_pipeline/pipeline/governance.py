from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any

from thesis_pipeline.stage_registry import StageID


REQUIRED_RESEARCH_DOCS = (
    "hypotheses.json",
    "success_criteria.md",
    "research_questions.md",
    "novelty_claim.md",
    "literature_gap_analysis.md",
)


def _as_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    return default


def get_pipeline_runtime_policy(config: Any) -> dict[str, Any]:
    pipeline_cfg = config.get("pipeline", {}) if hasattr(config, "get") else {}
    strict_fail = _as_bool(pipeline_cfg.get("strict_fail_policy", False), False)
    resume_enabled = _as_bool(pipeline_cfg.get("resume_enabled", False), False)

    strict_mode_cfg = pipeline_cfg.get("strict_mode", {}) if hasattr(pipeline_cfg, "get") else {}
    strict_mode_enabled = _as_bool(strict_mode_cfg.get("enabled", strict_fail), strict_fail)

    stage_toggles_raw = strict_mode_cfg.get("stage_toggles", {}) if hasattr(strict_mode_cfg, "get") else {}
    stage_toggles: dict[str, bool] = {}
    if isinstance(stage_toggles_raw, dict):
        for stage_id, enabled in stage_toggles_raw.items():
            stage_toggles[str(stage_id).upper()] = _as_bool(enabled, True)

    preflight_cfg = pipeline_cfg.get("preflight", {}) if hasattr(pipeline_cfg, "get") else {}
    preflight_enabled = _as_bool(preflight_cfg.get("enabled", True), True)
    require_research_docs = _as_bool(preflight_cfg.get("require_research_docs", False), False)
    require_europeana_key = _as_bool(preflight_cfg.get("require_europeana_api_key", True), True)
    require_git_clean = _as_bool(preflight_cfg.get("require_git_clean", False), False)

    return {
        "strict_fail_policy": strict_fail,
        "resume_enabled": resume_enabled,
        "strict_mode_enabled": strict_mode_enabled,
        "stage_toggles": stage_toggles,
        "preflight_enabled": preflight_enabled,
        "require_research_docs": require_research_docs,
        "require_europeana_api_key": require_europeana_key,
        "require_git_clean": require_git_clean,
    }


def run_preflight_checks(config: Any, *, project_root: Path, stages: list[StageID]) -> list[str]:
    policy = get_pipeline_runtime_policy(config)
    if not policy["preflight_enabled"]:
        return []

    errors: list[str] = []
    stage_ids = {stage.value for stage in stages}
    has_research_design = "S01" in stage_ids

    if has_research_design and policy["require_research_docs"]:
        docs_root = project_root / "docs" / "research_design"
        missing_docs = [name for name in REQUIRED_RESEARCH_DOCS if not (docs_root / name).exists()]
        if missing_docs:
            errors.append(
                "missing required research-design docs: " + ", ".join(sorted(missing_docs))
            )

    data_acq_cfg = config.get("data_acquisition", {}) if hasattr(config, "get") else {}
    europeana_enabled = _as_bool(data_acq_cfg.get("europeana_enabled", True), True)
    if europeana_enabled and policy["require_europeana_api_key"]:
        env_key = (os.environ.get("EUROPEANA_API_KEY") or "").strip()
        if not env_key:
            errors.append("EUROPEANA_API_KEY is required when europeana_enabled=true")

    if policy["require_git_clean"]:
        try:
            result = subprocess.run(
                ["git", "-C", str(project_root), "status", "--porcelain", "--untracked-files=no"],
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            if result.returncode != 0:
                detail = (result.stderr or result.stdout or "").strip()
                errors.append(f"git clean preflight failed: {detail or 'git status returned non-zero'}")
            else:
                dirty_entries = [line.strip() for line in result.stdout.splitlines() if line.strip()]
                if dirty_entries:
                    preview = ", ".join(dirty_entries[:10])
                    if len(dirty_entries) > 10:
                        preview += ", ..."
                    errors.append(f"git working tree is dirty: {preview}")
        except Exception as exc:
            errors.append(f"git clean preflight failed: {exc}")

    return errors
