import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class RealismValidator:
    """
    Validates synthetic damage against real damage patterns using geometric and texture proxies.
    """
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def analyze_mask_geometry(self, masks_dir: Path, name: str):
        """
        Analyzes edge complexity and roughness of masks.
        """
        mask_files = list(masks_dir.glob("*.png"))
        if not mask_files:
            return

        stats = []
        for f in mask_files:
            mask = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
            if mask is None: continue
            
            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < 10: continue
                
                perimeter = cv2.arcLength(cnt, True)
                # Complexity: Perimeter / Area ratio (higher = more irregular)
                complexity = perimeter / (area ** 0.5) if area > 0 else 0
                
                # Convexity: Area / Convex Hull Area
                hull = cv2.convexHull(cnt)
                hull_area = cv2.contourArea(hull)
                convexity = area / hull_area if hull_area > 0 else 1
                
                stats.append({
                    "filename": f.name,
                    "area": area,
                    "perimeter": perimeter,
                    "complexity": complexity,
                    "convexity": convexity
                })

        df = pd.DataFrame(stats)
        df.to_csv(self.output_dir / f"mask_geometry_{name}.csv", index=False)
        
        # Summary report
        summary = df.describe()
        summary.to_csv(self.output_dir / f"mask_geometry_summary_{name}.csv")
        
        logger.info(f"Realism validation (geometry) saved to {self.output_dir}")
        return df
