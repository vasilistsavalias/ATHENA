from thesis_pipeline.stages.stage_07_caption_generation import CaptionGenerationStage


def test_token_set_normalizes_words():
    tokens = CaptionGenerationStage._token_set("Crack on upper-left rim; missing fragment near base.")
    assert "crack" in tokens
    assert "upper" in tokens
    assert "rim" in tokens
    assert "base" in tokens


