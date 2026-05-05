import pandas as pd

from thesis_pipeline.stages.stage_11_feature_engineering import MaskRealismStage


def test_guardrails_pass_on_balanced_distribution():
    rows = []
    for mtype in ["rect", "irregular", "edge"]:
        for _ in range(20):
            rows.append({"coverage_ratio": 0.28, "mask_type": mtype})
    df = pd.DataFrame(rows)

    result = MaskRealismStage.evaluate_guardrails(df)

    assert result["passed"] is True
    assert result["status"] == "ok"


def test_guardrails_fail_on_extreme_coverage_and_imbalance():
    rows = []
    rows.extend({"coverage_ratio": 0.60, "mask_type": "rect"} for _ in range(50))
    rows.extend({"coverage_ratio": 0.05, "mask_type": "edge"} for _ in range(5))
    df = pd.DataFrame(rows)

    result = MaskRealismStage.evaluate_guardrails(df)

    assert result["passed"] is False
    assert result["status"] == "violations"
    assert any("median coverage" in r or "p90 coverage" in r for r in result["reasons"])


def test_guardrails_prefer_foreground_coverage_when_available():
    rows = []
    for mtype in ["rect", "irregular", "edge"]:
        for _ in range(20):
            rows.append(
                {
                    "coverage_ratio": 0.16,
                    "fg_coverage_ratio": 0.28,
                    "mask_type": mtype,
                }
            )
    df = pd.DataFrame(rows)

    result = MaskRealismStage.evaluate_guardrails(df)

    assert result["passed"] is True
    assert result["status"] == "ok"
    assert result["coverage_basis"] == "fg_coverage_ratio"


