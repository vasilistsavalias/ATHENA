from thesis_pipeline.stages.stage_07_caption_generation import _extract_spatial_caption


def test_extract_spatial_caption_prefers_spatial_chunks():
    text = (
        "A painted amphora with geometric motifs. "
        "A crack is visible on the upper-left rim and the missing fragment is near the base."
    )
    out = _extract_spatial_caption(text)
    assert "upper-left" in out.lower() or "upper" in out.lower()
    assert "base" in out.lower()


def test_extract_spatial_caption_falls_back_to_original_when_no_keywords():
    text = "A terracotta vessel with painted motifs and figures."
    out = _extract_spatial_caption(text)
    assert out == text


