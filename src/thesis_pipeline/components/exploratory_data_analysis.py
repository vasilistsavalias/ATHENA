# src/thesis_pipeline/components/exploratory_data_analysis.py
import logging
import pandas as pd
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from thesis_pipeline.visualization import ThesisPlotter

class ExploratoryDataAnalyzer:
    """
    Encapsulates the logic for performing EDA on a directory of images.
    Uses ThesisPlotter for publication-quality visualizations.
    """
    def __init__(self, input_dir: Path, output_dir: Path, extensions: list):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.extensions = extensions
        self.logger = logging.getLogger(__name__)
        self.df = None
        self.plotter = ThesisPlotter(output_dir)

    def _get_image_files(self) -> list:
        """Finds all image files with given extensions in the input directory.

        Uses non-recursive glob to avoid counting files in subdirectories
        (e.g. captions/ or refined_captions/) which would inflate the total.
        """
        image_files = []
        for ext in self.extensions:
            image_files.extend(self.input_dir.glob(f'*{ext}'))
        self.logger.info(f"Found {len(image_files)} image files in {self.input_dir}.")
        return image_files

    # ------------------------------------------------------------------
    # 3-Phase Validation (non-destructive sanity checks)
    # ------------------------------------------------------------------
    def validate_dataset(self, image_files: list) -> list:
        """
        Run three validation passes on the dataset.  Never deletes files;
        logs warnings for any anomalies and returns the list of files that
        pass all checks.

        Phase 1 – Readability: can Pillow open the file?
        Phase 2 – Color sanity: flag grayscale / book-scan candidates
        Phase 3 – Dimension sanity: flag very small or extreme-AR images
        """
        SATURATION_THRESHOLD = 25  # 0-255
        WHITENESS_THRESHOLD = 0.60
        MIN_DIM = 64
        AR_LOW, AR_HIGH = 0.3, 3.0

        valid_files = []
        phase1_fail = 0
        phase2_warn = 0
        phase3_warn = 0

        for img_path in tqdm(image_files, desc="Validating Dataset (3-phase)"):
            # ---- Phase 1: Readability ----
            try:
                with Image.open(img_path) as img:
                    width, height = img.size
                    mode = img.mode

                    # ---- Phase 2: Color sanity ----
                    if mode == 'L':
                        self.logger.warning(f"[Phase2] Grayscale image: {img_path.name}")
                        phase2_warn += 1
                    else:
                        img_hsv = img.convert('HSV')
                        np_img = np.array(img_hsv)
                        mean_sat = float(np.mean(np_img[:, :, 1]))
                        brightness = np_img[:, :, 2]
                        white_ratio = float(np.sum((brightness > 240) & (np_img[:, :, 1] < 30)) / brightness.size)

                        if mean_sat < SATURATION_THRESHOLD and white_ratio > WHITENESS_THRESHOLD:
                            self.logger.warning(
                                f"[Phase2] Possible book-scan/sketch: {img_path.name} "
                                f"(sat={mean_sat:.1f}, white={white_ratio:.2f})"
                            )
                            phase2_warn += 1

                    # ---- Phase 3: Dimension sanity ----
                    ar = width / height if height > 0 else 0
                    if width < MIN_DIM or height < MIN_DIM:
                        self.logger.warning(f"[Phase3] Tiny image: {img_path.name} ({width}x{height})")
                        phase3_warn += 1
                    elif ar < AR_LOW or ar > AR_HIGH:
                        self.logger.warning(f"[Phase3] Extreme AR: {img_path.name} (AR={ar:.2f})")
                        phase3_warn += 1

                    valid_files.append(img_path)

            except Exception as e:
                self.logger.warning(f"[Phase1] Unreadable image: {img_path.name} — {e}")
                phase1_fail += 1

        self.logger.info(
            f"3-Phase Validation: {len(valid_files)} passed, "
            f"{phase1_fail} unreadable, {phase2_warn} color warnings, "
            f"{phase3_warn} dimension warnings"
        )
        return valid_files

    def analyze_images(self, image_files: list):
        """Analyzes a list of image files and creates a DataFrame with metadata."""
        data = []
        self.logger.info("Analyzing image metadata...")
        for img_path in tqdm(image_files, desc="Analyzing Images"):
            try:
                with Image.open(img_path) as img:
                    width, height = img.size
                    data.append({
                        'filename': img_path.name,
                        'width': width,
                        'height': height,
                        'aspect_ratio': width / height if height > 0 else 0,
                        'mode': img.mode,
                        'filesize_kb': img_path.stat().st_size / 1024
                    })
            except Exception as e:
                self.logger.warning(f"Could not analyze image {img_path}. Error: {e}")
        
        self.df = pd.DataFrame(data)

    def generate_visualizations(self):
        """Generates and saves plots based on the image analysis DataFrame."""
        if self.df is None or self.df.empty:
            self.logger.warning("DataFrame is empty. Skipping visualization generation.")
            return

        self.logger.info("Generating thesis-quality visualizations...")

        # 1. Width Distribution
        self.plotter.plot_histogram(
            data=self.df['width'],
            title="Distribution of Image Widths",
            xlabel="Width (pixels)",
            filename="width_distribution",
            color='primary'
        )

        # 2. Height Distribution
        self.plotter.plot_histogram(
            data=self.df['height'],
            title="Distribution of Image Heights",
            xlabel="Height (pixels)",
            filename="height_distribution",
            color='secondary'
        )

        # 3. Scatter (Width vs Height)
        self.plotter.plot_scatter(
            x=self.df['width'],
            y=self.df['height'],
            title="Image Resolution Analysis",
            xlabel="Width (pixels)",
            ylabel="Height (pixels)",
            filename="resolution_scatter"
        )
        
        # 4. Aspect Ratio Distribution
        self.plotter.plot_histogram(
            data=self.df['aspect_ratio'],
            title="Distribution of Aspect Ratios",
            xlabel="Aspect Ratio (Width/Height)",
            filename="aspect_ratio_distribution",
            color='accent'
        )

        # 5. Sample Grid
        self._generate_sample_grid()

        self.logger.info(f"All visualizations saved to: {self.output_dir}")

    def _generate_sample_grid(self, num_samples=25):
        """Creates a grid of random images from the dataset."""
        try:
            sample_df = self.df.sample(n=min(num_samples, len(self.df)), random_state=42)
            images = []
            titles = []
            
            for _, row in sample_df.iterrows():
                img_path = self.input_dir / row['filename']
                if not img_path.exists(): continue
                
                with Image.open(img_path) as img:
                    img = img.convert('RGB')
                    img = img.resize((256, 256)) # Resize for uniform grid
                    images.append(np.array(img))
                    titles.append(f"{row['width']}x{row['height']}")
            
            self.plotter.plot_image_grid(
                images=images, 
                titles=titles, 
                rows=5, 
                cols=5, 
                filename="raw_data_samples"
            )
        except Exception as e:
            self.logger.warning(f"Failed to generate sample grid: {e}")

    def save_summary(self):
        """Saves summary statistics and the full metadata CSV."""
        if self.df is None or self.df.empty:
            self.logger.warning("DataFrame is empty. Skipping summary generation.")
            return
            
        # Summary Statistics
        summary_stats = self.df.describe()
        summary_path = self.output_dir / 'summary_statistics.txt'
        with open(summary_path, 'w') as f:
            f.write(f"Total Images Analyzed: {len(self.df)}\n\n{summary_stats.to_string()}")
        self.logger.info(f"Summary statistics saved to {summary_path}")

        # Full Metadata CSV
        csv_path = self.output_dir / 'full_image_metadata.csv'
        self.df.to_csv(csv_path, index=False)
        self.logger.info(f"Full metadata saved to {csv_path}")

    def run(self):
        """Executes the full EDA process: validate → analyze → visualize → save."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        image_files = self._get_image_files()
        
        if not image_files:
            self.logger.warning("No image files found. EDA concluded.")
            return
        
        # 3-phase validation (non-destructive, logs warnings only)
        image_files = self.validate_dataset(image_files)
        
        if not image_files:
            self.logger.warning("No valid image files after validation. EDA concluded.")
            return

        self.analyze_images(image_files)
        self.generate_visualizations()
        self.save_summary()
