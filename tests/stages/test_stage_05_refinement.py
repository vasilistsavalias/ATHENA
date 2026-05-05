import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest
from box import ConfigBox

if os.environ.get("TORCH_IMPORT_OK", "1") != "1":
    pytest.skip(
        "Skipping torch-dependent tests because torch import failed/crashed in a subprocess probe. "
        "Fix your torch install or set TORCH_IMPORT_OK=1 to force-run.",
        allow_module_level=True,
    )

pytest.importorskip("torch")

from thesis_pipeline.stages.stage_08_caption_refinement import CaptionRefinementStage


class _ConfigManager:
    def __init__(self, tmp_path: Path):
        self.config = ConfigBox(
            {
                "paths": {
                    "data": {
                        "filtered": str(tmp_path / "filtered"),
                    },
                    "artifacts": {
                        "stage_06": str(tmp_path / "stage_06"),
                        "stage_07": str(tmp_path / "stage_07"),
                    },
                },
                "caption_refinement": {
                    "mock": False,
                    "model_name": "test-model",
                    "max_tokens": 16,
                    "max_worker_fail_streak": 3,
                    "success_rate_min": 0.5,
                    "failure_policy": "hard_fail",
                },
            }
        )


def test_refinement_execution(tmp_path: Path):
    filtered_caption_dir = tmp_path / "filtered" / "accepted" / "captions"
    filtered_caption_dir.mkdir(parents=True, exist_ok=True)
    (filtered_caption_dir / "cap1.txt").write_text("caption one", encoding="utf-8")
    (filtered_caption_dir / "cap2.txt").write_text("caption two", encoding="utf-8")

    stage06_dir = tmp_path / "stage_06"
    stage06_dir.mkdir(parents=True, exist_ok=True)
    (stage06_dir / "caption_generation_report.json").write_text(
        json.dumps({"input_image_count": 2, "success_rate": 1.0}, indent=2),
        encoding="utf-8",
    )

    stage07_dir = tmp_path / "stage_07"
    stage07_dir.mkdir(parents=True, exist_ok=True)

    def _fake_gpu_parallel(task_func, data, **kwargs):
        output_dir = Path(kwargs["config_dict"]["output_dir"])
        for cap_file in data:
            (output_dir / cap_file.name).write_text("refined caption", encoding="utf-8")
        (output_dir / "worker_0_refine_report.json").write_text(
            json.dumps(
                {
                    "worker_id": 0,
                    "device": "cuda:0",
                    "model_name": "test-model",
                    "total_inputs": len(data),
                    "processed_count": len(data),
                    "failure_count": 0,
                    "model_load_error": None,
                    "aborted_due_to_fail_streak": False,
                    "max_fail_streak_observed": 0,
                    "failures": [],
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    stage = CaptionRefinementStage(_ConfigManager(tmp_path))

    with patch(
        "thesis_pipeline.stages.stage_08_caption_refinement.ParallelExecutor.run_gpu_parallel",
        side_effect=_fake_gpu_parallel,
    ) as mock_parallel:
        stage.run()

    mock_parallel.assert_called_once()
    report_path = stage.artifacts_dir / "caption_refinement_report.json"
    assert report_path.exists()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["refined_caption_count"] == 2
    assert report["missing_or_failed_count"] == 0


