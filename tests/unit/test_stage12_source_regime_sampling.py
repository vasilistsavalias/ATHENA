from pathlib import Path

from thesis_pipeline.stages.stage_13_model_training import ModelTrainingStage


class _DummyDataset:
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)


def _sample(stem: str):
    p = Path(stem + ".png")
    return (p, p, p)


def test_source_regime_sampling_biased_ratio_control(tmp_path):
    stage = ModelTrainingStage.__new__(ModelTrainingStage)
    stage.global_seed = 42

    wiki = [_sample(f"wiki_{i:03d}") for i in range(600)]
    eur = [_sample(f"eur_{i:03d}") for i in range(10)]
    ds = _DummyDataset(wiki + eur)

    subset, provenance = stage._build_source_regime_subset(
        ds,
        regime_mode="biased",
        report_root=tmp_path,
        source_cfg={"wikimedia_to_europeana_ratio": 261, "seed": 42},
    )

    assert len(subset) > 0
    assert provenance["regime_mode"] == "biased"
    assert provenance["source_counts_selected"]["wikimedia"] >= provenance["source_counts_selected"]["europeana"]


def test_source_regime_sampling_balanced_equalizes_sources(tmp_path):
    stage = ModelTrainingStage.__new__(ModelTrainingStage)
    stage.global_seed = 42

    wiki = [_sample(f"wiki_{i:03d}") for i in range(50)]
    eur = [_sample(f"eur_{i:03d}") for i in range(30)]
    ds = _DummyDataset(wiki + eur)

    subset, provenance = stage._build_source_regime_subset(
        ds,
        regime_mode="balanced",
        report_root=tmp_path,
        source_cfg={"wikimedia_to_europeana_ratio": 261, "seed": 42},
    )

    assert len(subset) > 0
    assert provenance["regime_mode"] == "balanced"
    assert provenance["source_counts_selected"]["wikimedia"] == provenance["source_counts_selected"]["europeana"]


