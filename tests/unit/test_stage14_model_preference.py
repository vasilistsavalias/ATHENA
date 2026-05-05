from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from box import ConfigBox

from thesis_pipeline.stages.stage_16_deployment_preparation import DeploymentPreparationStage


class _DummyConfigManager:
    def __init__(self, root: Path):
        self._stage_12 = root / "stage11"
        self._stage_16 = root / "stage16"
        self._paths = SimpleNamespace(
            data=SimpleNamespace(models=str(root / "models")),
            artifacts=SimpleNamespace(stage_11=str(root / "stage11"), stage_14=str(root / "stage14")),
        )
        self.config = ConfigBox({"deployment_preparation": {"export_formats": ["pt"]}})

    def get_paths(self):
        return self._paths

    def get_stage_artifact_dir(self, stage_id: str) -> Path:
        if stage_id == "S12":
            return self._stage_12
        if stage_id == "S16":
            return self._stage_16
        raise KeyError(stage_id)

    def get_stage_artifact_path(self, stage_id: str, *parts: str) -> Path:
        return self.get_stage_artifact_dir(stage_id).joinpath(*parts)


class _FakePackager:
    last_args = None

    def __init__(self, cfg, model_input, hyperparams_input, output_dir):
        _FakePackager.last_args = {
            "model_input": Path(model_input),
            "hyperparams_input": Path(hyperparams_input),
            "output_dir": Path(output_dir),
        }

    def package(self):
        return None


def test_stage14_prefers_unet_best(monkeypatch, tmp_path):
    cm = _DummyConfigManager(tmp_path)
    models = Path(cm.get_paths().data.models)
    stage11 = Path(cm.get_paths().artifacts.stage_11)
    (models / "unet_best").mkdir(parents=True, exist_ok=True)
    (models / "unet_final").mkdir(parents=True, exist_ok=True)
    stage11.mkdir(parents=True, exist_ok=True)
    (stage11 / "best_hyperparameters.yaml").write_text("learning_rate: 1e-5\n", encoding="utf-8")

    monkeypatch.setattr(
        "thesis_pipeline.stages.stage_16_deployment_preparation.DeploymentPackager",
        _FakePackager,
    )

    stage = DeploymentPreparationStage(cm)
    stage.run()

    assert _FakePackager.last_args is not None
    assert _FakePackager.last_args["model_input"].name == "unet_best"


