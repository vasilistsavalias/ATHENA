from pathlib import Path

import yaml

from thesis_pipeline.config_manager import ConfigManager


def test_config_manager_supports_extends_and_artifact_root_rewrite(tmp_path: Path):
    base_config = {
        "paths": {
            "artifacts": {
                "root": "outputs",
                "S00": "outputs/S00_reproducibility",
                "S15": "outputs/S15_model_evaluation",
                "S18": "outputs/S18_expert_validation",
                "logs": "outputs/00_logs",
            }
        },
        "pipeline": {"resume_enabled": True},
    }
    child_config = {
        "extends": "base.yaml",
        "pipeline": {"resume_enabled": False},
    }

    base_path = tmp_path / "base.yaml"
    child_path = tmp_path / "child.yaml"
    base_path.write_text(yaml.safe_dump(base_config), encoding="utf-8")
    child_path.write_text(yaml.safe_dump(child_config), encoding="utf-8")

    cm = ConfigManager(config_filepath=child_path)
    assert cm.config.pipeline.resume_enabled is False
    assert cm.get_stage_artifact_dir("S15") == Path("outputs/S15_model_evaluation")

    cm.rewrite_artifacts_root(tmp_path / "custom_outputs")
    assert cm.get_stage_artifact_dir("S15") == tmp_path / "custom_outputs" / "S15_model_evaluation"
    assert cm.get_stage_artifact_dir("S18") == tmp_path / "custom_outputs" / "S18_expert_validation"
    assert Path(cm.get_paths().artifacts.logs) == tmp_path / "custom_outputs" / "00_logs"
