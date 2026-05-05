# src/thesis_pipeline/components/processing.py
import logging
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from thesis_pipeline.visualization import ThesisPlotter

class ImageProcessor:
    """
    Encapsulates the logic for processing raw images.
    """
    def __init__(self, input_dir: Path, output_dir: Path, report_dir: Path, image_size: tuple, output_format: str):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.target_size = image_size
        self.output_format = output_format
        self.logger = logging.getLogger(__name__)
        self.plotter = ThesisPlotter(report_dir)

    def _find_image_files(self) -> list:
        """Finds all image files in the input directory."""
        extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
        image_files = []
        for ext in extensions:
            image_files.extend(self.input_dir.glob(f'*{ext}'))
        image_files = sorted(image_files, key=lambda p: p.name)
        self.logger.info(f"Found {len(image_files)} image files in {self.input_dir}.")
        return image_files

    def process_images(self, max_images=None) -> dict:
        """
        Processes raw images by resizing, converting to RGB, and saving them.
        Returns a summary of the operation.
        """
        image_files = self._find_image_files()
        total_found = len(image_files)

        if isinstance(max_images, int) and max_images > 0:
            image_files = image_files[:max_images]
            self.logger.info(f"Limiting processing to {len(image_files)} images (max_images={max_images}).")
        
        if not image_files:
            self.logger.warning("No images found in the input directory. Nothing to process.")
            return {"processed_count": 0, "error_count": 0, "total_found": 0}

        self.logger.info(f"Processing {len(image_files)} images. Target size: {self.target_size}, Format: {self.output_format}")
        
        processed_count = 0
        error_count = 0
        sample_images = []

        for img_path in tqdm(image_files, desc="Processing Images"):
            try:
                with Image.open(img_path) as img:
                    img_rgb = img.convert('RGB')
                    
                    # --- Resize with Padding (Letterbox) ---
                    target_w, target_h = self.target_size
                    original_w, original_h = img_rgb.size
                    
                    # Calculate scale factor
                    scale = min(target_w / original_w, target_h / original_h)
                    new_w = int(original_w * scale)
                    new_h = int(original_h * scale)
                    
                    # Resize preserving aspect ratio
                    img_resized = img_rgb.resize((new_w, new_h), Image.Resampling.LANCZOS)
                    
                    # Create black canvas
                    new_img = Image.new("RGB", self.target_size, (0, 0, 0))
                    
                    # Paste resized image in center
                    paste_x = (target_w - new_w) // 2
                    paste_y = (target_h - new_h) // 2
                    new_img.paste(img_resized, (paste_x, paste_y))
                    
                    # Save
                    output_filename = f"{img_path.stem}.{self.output_format.lower()}"
                    output_path = self.output_dir / output_filename
                    
                    new_img.save(output_path, format=self.output_format)
                    
                    # Copy metadata if exists
                    metadata_path = img_path.parent / "metadata" / f"{img_path.name}.txt"
                    if metadata_path.exists():
                        import shutil
                        shutil.copy(metadata_path, self.output_dir / f"{img_path.stem}.txt")

                    # Copy caption: prefer refined (Stage 07), fall back to raw (Stage 06)
                    import shutil as _shutil
                    refined_caption = img_path.parent / "refined_captions" / f"{img_path.stem}.txt"
                    raw_caption = img_path.parent / "captions" / f"{img_path.stem}.txt"
                    caption_src = refined_caption if refined_caption.exists() else raw_caption
                    if caption_src.exists():
                        _shutil.copy(caption_src, self.output_dir / f"{img_path.stem}.txt")
                        
                    processed_count += 1
                    
                    # Collect samples for visualization
                    if len(sample_images) < 25:
                        sample_images.append(np.array(new_img))

            except Exception as e:
                self.logger.error(f"Failed to process {img_path}. Error: {e}")
                error_count += 1
        
        # Generate Visualization
        if sample_images:
            self.plotter.plot_image_grid(sample_images, None, 5, 5, "processed_samples_grid")
        
        summary = {
            "processed_count": processed_count,
            "error_count": error_count,
            "total_found": total_found,
            "selected_count": len(image_files),
        }
        self.logger.info(f"Image processing complete. Summary: {summary}")
        return summary

