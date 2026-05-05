from thesis_pipeline.stages.stage_02_data_acquisition import DataAcquisitionStage


def test_stage02_validity_detects_zero_enabled_source():
    violations = DataAcquisitionStage._evaluate_validity_violations(
        counts_after={"wikimedia": 10, "europeana": 0},
        total_after=10,
        wiki_enabled=True,
        eur_enabled=True,
        min_total_images=0,
        require_enabled_source_nonzero=True,
        require_europeana_key_when_enabled=False,
        europeana_key_present=False,
    )

    assert any("europeana_enabled=true but downloaded count is zero" in v for v in violations)


def test_stage02_validity_detects_min_total_floor():
    violations = DataAcquisitionStage._evaluate_validity_violations(
        counts_after={"wikimedia": 15, "europeana": 5},
        total_after=20,
        wiki_enabled=True,
        eur_enabled=True,
        min_total_images=100,
        require_enabled_source_nonzero=True,
        require_europeana_key_when_enabled=False,
        europeana_key_present=True,
    )

    assert any("below configured minimum" in v for v in violations)


def test_stage02_validity_requires_key_when_enabled():
    violations = DataAcquisitionStage._evaluate_validity_violations(
        counts_after={"wikimedia": 30, "europeana": 20},
        total_after=50,
        wiki_enabled=True,
        eur_enabled=True,
        min_total_images=0,
        require_enabled_source_nonzero=False,
        require_europeana_key_when_enabled=True,
        europeana_key_present=False,
    )

    assert any("EUROPEANA_API_KEY" in v for v in violations)

