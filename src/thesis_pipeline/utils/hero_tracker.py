import logging
import shutil
import json
from pathlib import Path
import cv2
import numpy as np

class HeroTracker:
    """
    Tracks specific 'hero' samples through the pipeline stages.
    """
    def __init__(self, output_dir: Path, hero_filenames: list = None):
        self.output_dir = output_dir
        self.hero_filenames = hero_filenames or []
        self.logger = logging.getLogger(__name__)
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def is_hero(self, filename: str) -> bool:
        if not self.hero_filenames:
            return False # Or True if we want to track everything (bad idea)
        
        # Check if filename contains any hero identifier (stem match)
        stem = Path(filename).stem
        for hero in self.hero_filenames:
            if hero in stem:
                return True
        return False

    def log_image(self, image, stage_name: str, filename: str, caption: str = None):
        if not self.is_hero(filename):
            return

        stem = Path(filename).stem
        hero_dir = self.output_dir / stem
        hero_dir.mkdir(parents=True, exist_ok=True)
        
        # Save Image
        save_path = hero_dir / f"{stage_name}.png"
        
        if isinstance(image, (str, Path)):
            shutil.copy(image, save_path)
        elif isinstance(image, np.ndarray):
            cv2.imwrite(str(save_path), image)
        else:
            # Assume PIL
            image.save(save_path)
            
        # Save Caption/Metadata if provided
        if caption:
            with open(hero_dir / f"{stage_name}_meta.txt", "w", encoding="utf-8") as f:
                f.write(caption)
        
        self.logger.info(f"Logged hero state for {stem} at stage {stage_name}")

    def log_text(self, text: str, stage_name: str, filename: str):
        if not self.is_hero(filename):
            return

        stem = Path(filename).stem
        hero_dir = self.output_dir / stem
        hero_dir.mkdir(parents=True, exist_ok=True)
        
        with open(hero_dir / f"{stage_name}.txt", "w", encoding="utf-8") as f:
            f.write(text)
        self.logger.info(f"Logged hero text for {stem} at stage {stage_name}")
