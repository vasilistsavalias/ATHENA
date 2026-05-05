from pathlib import Path

import subprocess
from thesis_pipeline.pipeline.governance import get_pipeline_runtime_policy, run_preflight_checks
from thesis_pipeline.stage_registry import StageID


def test_runtime_policy_parses_strict_mode_and_toggles():
    cfg = {
        "pipeline": {
            "strict_fail_policy": True,
            "resume_enabled": False,
            "strict_mode": {
                "enabled": True,
                "stage_toggles": {"S01": True, "S02": False},
            },
            "preflight": {
                "enabled": True,
                "require_research_docs": False,
                "require_europeana_api_key": True,
            },
        }
    }

    policy = get_pipeline_runtime_policy(cfg)
    assert policy["strict_fail_policy"] is True
    assert policy["strict_mode_enabled"] is True
    assert policy["stage_toggles"]["S01"] is True
    assert policy["stage_toggles"]["S02"] is False


def test_preflight_reports_missing_research_docs(tmp_path, monkeypatch):
    project_root = Path(tmp_path)
    (project_root / "docs").mkdir(parents=True, exist_ok=True)

    cfg = {
        "pipeline": {
            "preflight": {
                "enabled": True,
                "require_research_docs": True,
                "require_europeana_api_key": False,
            }
        },
        "data_acquisition": {"europeana_enabled": False},
    }

    errors = run_preflight_checks(cfg, project_root=project_root, stages=[StageID.S01])
    assert errors
    assert any("missing required research-design docs" in msg for msg in errors)


def test_preflight_does_not_require_research_docs_by_default(tmp_path):
    cfg = {
        "pipeline": {
            "preflight": {
                "enabled": True,
                "require_europeana_api_key": False,
            }
        },
        "data_acquisition": {"europeana_enabled": False},
    }

    errors = run_preflight_checks(cfg, project_root=Path(tmp_path), stages=[StageID.S01])
    assert not any("missing required research-design docs" in msg for msg in errors)


def test_preflight_requires_europeana_key_when_enabled(tmp_path, monkeypatch):
    monkeypatch.delenv("EUROPEANA_API_KEY", raising=False)

    cfg = {
        "pipeline": {
            "preflight": {
                "enabled": True,
                "require_research_docs": False,
                "require_europeana_api_key": True,
            }
        },
        "data_acquisition": {"europeana_enabled": True},
    }

    errors = run_preflight_checks(cfg, project_root=Path(tmp_path), stages=[StageID.S01])
    assert any("EUROPEANA_API_KEY" in msg for msg in errors)


def test_preflight_requires_clean_git_tree(tmp_path, monkeypatch):
    cfg = {
        "pipeline": {
            "preflight": {
                "enabled": True,
                "require_research_docs": False,
                "require_europeana_api_key": False,
                "require_git_clean": True,
            }
        },
        "data_acquisition": {"europeana_enabled": False},
    }

    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(
            args=args[0],
            returncode=0,
            stdout=" M src/thesis_pipeline/pipeline/app.py\n",
            stderr="",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    errors = run_preflight_checks(cfg, project_root=Path(tmp_path), stages=[StageID.S01])
    assert any("git working tree is dirty" in msg for msg in errors)
