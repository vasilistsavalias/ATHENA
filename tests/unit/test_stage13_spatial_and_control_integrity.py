import json

import pytest

from thesis_pipeline.stages.stage_15_model_evaluation import ModelEvaluationStage


def test_expected_combinations_include_spatial_condition_when_enabled():
    combos = ModelEvaluationStage._expected_combinations(
        has_finetuned_model=True,
        deep_enabled=False,
        include_spatial_condition=True,
    )
    assert ("Vanilla SD", "Spatial Damage Context") in combos
    assert ("FT-SD", "Spatial Damage Context") in combos


def test_expected_combinations_skip_spatial_condition_when_disabled():
    combos = ModelEvaluationStage._expected_combinations(
        has_finetuned_model=True,
        deep_enabled=False,
        include_spatial_condition=False,
    )
    assert ("Vanilla SD", "Spatial Damage Context") not in combos
    assert ("FT-SD", "Spatial Damage Context") not in combos


def test_spatial_condition_gate_disabled_when_grounding_fails(tmp_path):
    stage = ModelEvaluationStage.__new__(ModelEvaluationStage)
    stage.logger = type("L", (), {"warning": lambda *args, **kwargs: None})()
    stage.config = {
        "pipeline": {"strict_fail_policy": False},
        "model_evaluation": {"include_spatial_condition": True},
    }

    report_path = tmp_path / "stage_06b_grounding_validation.json"
    report_path.write_text(
        json.dumps({"pass": False, "quadrant_macro_f1": 0.2, "border_touch_accuracy": 0.1, "area_correlation": 0.0}),
        encoding="utf-8",
    )

    include = bool(stage.config["model_evaluation"].get("include_spatial_condition", True))
    grounding_report = json.loads(report_path.read_text(encoding="utf-8"))
    if include and not bool(grounding_report.get("pass", False)):
        include = False
    assert include is False


def test_spatial_condition_gate_raises_in_strict_mode(tmp_path):
    stage = ModelEvaluationStage.__new__(ModelEvaluationStage)
    stage.logger = type("L", (), {"warning": lambda *args, **kwargs: None})()
    stage.config = {
        "pipeline": {"strict_fail_policy": True},
        "model_evaluation": {"include_spatial_condition": True},
    }
    report_path = tmp_path / "stage_06b_grounding_validation.json"
    report_path.write_text(json.dumps({"pass": False}), encoding="utf-8")

    with pytest.raises(RuntimeError):
        include = bool(stage.config["model_evaluation"].get("include_spatial_condition", True))
        grounding_report = json.loads(report_path.read_text(encoding="utf-8"))
        if include and not bool(grounding_report.get("pass", False)):
            if bool(stage.config.get("pipeline", {}).get("strict_fail_policy", False)):
                raise RuntimeError("Spatial condition blocked by Stage06b grounding gate")


