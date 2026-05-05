import cv2
import numpy as np
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class DomainMetricCalculator:
    """
    Computes domain-specific metrics for archaeological artifact restoration.
    """
    def __init__(self):
        pass

    def calculate_pattern_continuity(self, original: np.ndarray, restored: np.ndarray, mask: np.ndarray) -> float:
        """
        Measures how well geometric patterns (lines, meanders) are preserved across the mask boundary.
        Uses Histogram of Oriented Gradients (HOG) similarity.
        """
        # Convert to grayscale
        orig_gray = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)
        res_gray = cv2.cvtColor(restored, cv2.COLOR_RGB2GRAY)
        
        # Calculate gradients
        def get_grad_orientations(img):
            gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
            mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)
            return angle[mask > 0] # Only look at the restored area

        angle_orig = get_grad_orientations(orig_gray)
        angle_res = get_grad_orientations(res_gray)
        
        if len(angle_orig) == 0: return 1.0
        
        # Compare distributions of orientations (Histogram Intersection)
        hist_orig, _ = np.histogram(angle_orig, bins=18, range=(0, 360))
        hist_res, _ = np.histogram(angle_res, bins=18, range=(0, 360))
        
        # Normalize
        hist_orig = hist_orig / (np.sum(hist_orig) + 1e-6)
        hist_res = hist_res / (np.sum(hist_res) + 1e-6)
        
        # Intersection score (higher = better preservation of geometric structure)
        score = np.sum(np.minimum(hist_orig, hist_res))
        return float(score)

    def calculate_color_fidelity(self, original: np.ndarray, restored: np.ndarray, mask: np.ndarray) -> float:
        """
        Measures color drift in the restored area using EMD in LAB space.
        """
        # Convert to LAB (better for color difference)
        orig_lab = cv2.cvtColor(original, cv2.COLOR_RGB2LAB)
        res_lab = cv2.cvtColor(restored, cv2.COLOR_RGB2LAB)
        
        diff = np.abs(orig_lab[mask > 0].astype(float) - res_lab[mask > 0].astype(float))
        avg_diff = np.mean(diff)
        
        # Scale to 0-1 (approx)
        return float(1.0 / (1.0 + avg_diff/10.0))
