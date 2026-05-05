import pandas as pd
import shutil
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class FailureAnalyzer:
    """
    Identifies and extracts samples where the model performed poorly.
    """
    def __init__(self, metrics_file: Path, output_dir: Path):
        self.metrics_file = metrics_file
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def extract_failures(self, sample_root: Path, top_n: int = 10, metric: str = 'psnr'):
        if not self.metrics_file.exists():
            return

        df = pd.read_csv(self.metrics_file)
        # Sort by metric (ascending for PSNR/SSIM, descending for LPIPS)
        ascending = True if metric in ['psnr', 'ssim'] else False
        worst = df.sort_values(by=metric, ascending=ascending).head(top_n)

        logger.info(f"Extracting top {top_n} failure cases based on {metric}...")

        failure_summary = []

        for _, row in worst.iterrows():
            fname = row['filename']
            stem = Path(fname).stem
            src_dir = sample_root / stem
            dst_dir = self.output_dir / stem
            
            if src_dir.exists():
                if dst_dir.exists(): shutil.rmtree(dst_dir)
                shutil.copytree(src_dir, dst_dir)
                
            failure_summary.append({
                "filename": fname,
                "score": row[metric],
                "difficulty": row.get('difficulty', 'Unknown'),
                "coverage": row.get('mask_coverage', 0)
            })

        pd.DataFrame(failure_summary).to_csv(self.output_dir / "failure_summary.csv", index=False)
        logger.info(f"Failures extracted to {self.output_dir}")
