import pandas as pd

from thesis_pipeline.analysis.bias_analyzer import BiasAnalyzer


def test_bias_disparity_computes_gap_between_sources():
    df = pd.DataFrame(
        {
            "image_id": ["wiki_a.jpg", "wiki_b.jpg", "eur_a.jpg", "eur_b.jpg"],
            "status": ["Accepted", "Accepted", "Rejected", "Accepted"],
        }
    )

    result = BiasAnalyzer.compute_source_disparity(df)
    assert result["available"] is True
    assert result["rates"]["wikimedia"] == 1.0
    assert result["rates"]["europeana"] == 0.5
    assert abs(result["absolute_gap"] - 0.5) < 1e-9


def test_bias_disparity_marks_unavailable_with_single_source():
    df = pd.DataFrame(
        {
            "image_id": ["wiki_a.jpg", "wiki_b.jpg"],
            "status": ["Accepted", "Rejected"],
        }
    )

    result = BiasAnalyzer.compute_source_disparity(df)
    assert result["available"] is False
    assert result["reason"] == "missing_one_source"
