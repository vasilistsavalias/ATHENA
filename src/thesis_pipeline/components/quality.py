import cv2
import numpy as np
import logging

class QualityFilter:
    def __init__(self, blur_threshold=100.0, blur_metric: str = "tenengrad", resize_max_dim: int = 512):
        self.blur_threshold = blur_threshold
        self.blur_metric = (blur_metric or "tenengrad").strip().lower()
        self.resize_max_dim = int(resize_max_dim) if resize_max_dim else 0
        self.logger = logging.getLogger(__name__)

    def _prepare_gray(self, image: np.ndarray) -> np.ndarray | None:
        if image is None:
            return None

        img = image
        if self.resize_max_dim and self.resize_max_dim > 0:
            h, w = img.shape[:2]
            m = max(h, w) if h and w else 0
            if m and m > self.resize_max_dim:
                scale = float(self.resize_max_dim) / float(m)
                img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    def compute_focus_score(self, image: np.ndarray) -> float:
        """Compute focus score; lower scores indicate blurrier crops."""
        gray = self._prepare_gray(image)
        if gray is None:
            return 0.0

        if self.blur_metric == "laplacian_var":
            return float(cv2.Laplacian(gray, cv2.CV_64F).var())

        gx = cv2.Scharr(gray, cv2.CV_64F, 1, 0)
        gy = cv2.Scharr(gray, cv2.CV_64F, 0, 1)
        return float((gx * gx + gy * gy).mean())

    def detect_blur(self, image: np.ndarray, threshold: float | None = None) -> tuple[bool, float]:
        """
        Detects if an image is blurry using a focus-measure metric.

        Metrics:
        - tenengrad (default): mean squared gradient magnitude (Scharr).
        - laplacian_var: variance of Laplacian (legacy).

        Returns: (is_blurry, score) where lower scores are blurrier.
        """
        score = self.compute_focus_score(image)
        active_threshold = self.blur_threshold if threshold is None else float(threshold)
        is_blurry = score < active_threshold
        return is_blurry, score
