import numpy as np
from PIL import Image

from thesis_pipeline.components.data_cleaning import DataCleaner


def _solid_rgb(width, height, rgb):
    arr = np.zeros((height, width, 3), dtype=np.uint8)
    arr[:, :] = np.array(rgb, dtype=np.uint8)
    return Image.fromarray(arr, mode="RGB")


def test_grayscale_mode_rejected():
    gray = Image.fromarray(np.full((300, 300), 128, dtype=np.uint8), mode="L")
    result = DataCleaner.evaluate_image_quality(
        img=gray,
        min_width=256,
        min_height=256,
        color_check=True,
        saturation_threshold=20.0,
        whiteness_threshold=0.85,
    )
    assert result["accepted"] is False
    assert result["reason"] == "Grayscale Mode"


def test_low_sat_high_white_rejected():
    # Near-white, low-saturation background.
    white_like = _solid_rgb(300, 300, (245, 245, 245))
    result = DataCleaner.evaluate_image_quality(
        img=white_like,
        min_width=256,
        min_height=256,
        color_check=True,
        saturation_threshold=20.0,
        whiteness_threshold=0.85,
    )
    assert result["accepted"] is False
    assert "Low Saturation + High Whiteness" in result["reason"]


def test_low_sat_but_not_high_white_accepted():
    # Dark gray gives low saturation but not high whiteness.
    dark_gray = _solid_rgb(300, 300, (80, 80, 80))
    result = DataCleaner.evaluate_image_quality(
        img=dark_gray,
        min_width=256,
        min_height=256,
        color_check=True,
        saturation_threshold=20.0,
        whiteness_threshold=0.85,
    )
    assert result["accepted"] is True
    assert result["reason"] == "Valid"

