from app.services.export_service import _build_reliability_report


def test_build_reliability_report_computes_block_metrics():
    item_rows = [
        {"block": "A", "item_id": 1, "authenticity_likelihood": 4, "archaeological_plausibility": 5},
        {"block": "A", "item_id": 1, "authenticity_likelihood": 5, "archaeological_plausibility": 4},
        {"block": "A", "item_id": 2, "authenticity_likelihood": 2, "archaeological_plausibility": 2},
        {"block": "A", "item_id": 2, "authenticity_likelihood": 5, "archaeological_plausibility": 5},
    ]
    pairwise_rows = [
        {"item_id": 10, "choice": "A"},
        {"item_id": 10, "choice": "A"},
        {"item_id": 11, "choice": "A"},
        {"item_id": 11, "choice": "B"},
    ]

    report = _build_reliability_report(item_rows, pairwise_rows)
    assert report["block_a"]["items_with_multiple_raters_authenticity"] == 2
    assert report["block_b"]["items_with_multiple_raters"] == 2
    assert report["block_b"]["mean_pairwise_percent_agreement"] is not None
