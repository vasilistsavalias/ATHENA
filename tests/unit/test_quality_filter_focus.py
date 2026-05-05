import numpy as np
import cv2

from thesis_pipeline.components.quality import QualityFilter


def _make_checkerboard(size=256, block=16):
    image = np.zeros((size, size), dtype=np.uint8)
    for y in range(0, size, block):
        for x in range(0, size, block):
            if ((x // block) + (y // block)) % 2 == 0:
                image[y : y + block, x : x + block] = 255
    return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)


def test_tenengrad_sharp_above_blur():
    sharp = _make_checkerboard()
    blurred = cv2.GaussianBlur(sharp, (21, 21), 0)

    quality_filter = QualityFilter(blur_threshold=4500.0, blur_metric="tenengrad", resize_max_dim=512)
    sharp_score = quality_filter.compute_focus_score(sharp)
    blurred_score = quality_filter.compute_focus_score(blurred)

    assert sharp_score > blurred_score


def test_focus_score_deterministic_with_resize():
    image = _make_checkerboard(size=1024, block=32)
    quality_filter = QualityFilter(blur_threshold=4500.0, blur_metric="tenengrad", resize_max_dim=512)

    score_1 = quality_filter.compute_focus_score(image)
    score_2 = quality_filter.compute_focus_score(image)

    assert score_1 == score_2

