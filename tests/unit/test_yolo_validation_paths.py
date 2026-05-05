import unittest
from pathlib import Path
import tempfile
import yaml
import os
import sys
import pytest

if os.environ.get("TORCH_IMPORT_OK", "1") != "1":
    pytest.skip(
        "Skipping torch-dependent tests because torch import failed/crashed in a subprocess probe. "
        "Fix your torch install or set TORCH_IMPORT_OK=1 to force-run.",
        allow_module_level=True,
    )

pytest.importorskip("torch")
from thesis_pipeline.config_manager import ConfigManager
from thesis_pipeline.stages.stage_04_yolo_validation import YOLOValidationStage


class TestYoloValidationPaths(unittest.TestCase):
    def test_ground_truth_path_from_config(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            gt_path = tmp_path / "data" / "labels" / "ground_truth_50.json"
            gt_path.parent.mkdir(parents=True, exist_ok=True)
            gt_path.write_text("{}", encoding="utf-8")

            config = {
                "paths": {
                    "data": {"raw": "data/01_raw"},
                    "artifacts": {"stage_03": "outputs/stage_03"},
                },
                "intelligent_filtering": {
                    "ground_truth": str(gt_path),
                    "model_name": "data/models/weights/yolov8n.pt",
                },
            }
            config_path = tmp_path / "config.yaml"
            config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

            cm = ConfigManager(config_filepath=config_path)
            stage = YOLOValidationStage(cm)
            resolved = stage._resolve_ground_truth_path()

            self.assertEqual(Path(resolved), gt_path)


if __name__ == "__main__":
    unittest.main()


