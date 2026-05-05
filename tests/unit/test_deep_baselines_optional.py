from thesis_pipeline.components.evaluation.deep_baselines import build_deep_baseline_runner


def test_deep_baseline_probe_has_expected_keys(tmp_path):
    runner, probe = build_deep_baseline_runner(cfg={"enabled": False}, stage13_dir=tmp_path)
    assert runner.enabled is False
    names = {p.name for p in probe}
    assert {"iopaint", "gdown", "git"}.issubset(names)
