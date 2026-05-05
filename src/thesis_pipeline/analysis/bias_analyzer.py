import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
import json

logger = logging.getLogger(__name__)

class BiasAnalyzer:
    """
    Analyzes the rejection log to detect sampling bias in the filtering process.
    """
    def __init__(self, log_file: Path, output_dir: Path):
        self.log_file = log_file
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _normalize_status(value: str) -> str:
        text = str(value or "").strip().lower()
        if text in {"accepted", "accept", "kept", "pass", "passed"}:
            return "accepted"
        return "rejected"

    @staticmethod
    def _infer_source(image_id: str) -> str:
        name = str(image_id or "").strip().lower()
        if name.startswith("wiki_"):
            return "wikimedia"
        if name.startswith("eur_"):
            return "europeana"
        return "unknown"

    @staticmethod
    def compute_source_disparity(df: pd.DataFrame) -> dict:
        work = df.copy()
        if "status" in work.columns:
            status_series = work["status"]
        else:
            status_series = pd.Series([""] * len(work), index=work.index)
        work["status_norm"] = status_series.map(BiasAnalyzer._normalize_status)

        if "image_id" in work.columns:
            source_seed = work["image_id"]
        elif "filename" in work.columns:
            source_seed = work["filename"]
        else:
            source_seed = pd.Series([""] * len(work), index=work.index)
        work["source"] = source_seed.map(BiasAnalyzer._infer_source)
        work = work[work["source"].isin(["wikimedia", "europeana"])]

        if work.empty:
            return {
                "available": False,
                "reason": "no_source_labeled_rows",
                "rates": {},
                "absolute_gap": None,
            }

        grouped = work.groupby("source")["status_norm"].apply(
            lambda s: float((s == "accepted").sum()) / float(len(s)) if len(s) else 0.0
        )
        rates = {k: float(v) for k, v in grouped.to_dict().items()}

        if "wikimedia" not in rates or "europeana" not in rates:
            return {
                "available": False,
                "reason": "missing_one_source",
                "rates": rates,
                "absolute_gap": None,
            }

        gap = abs(rates["wikimedia"] - rates["europeana"])
        return {
            "available": True,
            "reason": None,
            "rates": rates,
            "absolute_gap": float(gap),
        }

    def analyze(self):
        if not self.log_file.exists():
            logger.warning(f"Log file not found: {self.log_file}")
            return

        df = pd.read_csv(self.log_file)
        if df.empty:
            logger.warning("Log file is empty.")
            return

        logger.info(f"Analyzing bias from {len(df)} records...")

        # 1. Rejection Reasons Distribution
        plt.figure(figsize=(10, 6))
        sns.countplot(data=df, x='reason', hue='status')
        plt.title("Distribution of Rejection Reasons")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(self.output_dir / "rejection_reasons.png")
        plt.close()

        # 2. Focus/Blur score vs status (supports legacy and V7 schema)
        score_col = "focus_score" if "focus_score" in df.columns else "blur_variance"
        df_blur = df[df.get(score_col).notna()].copy() if score_col in df.columns else pd.DataFrame()
        if not df_blur.empty:
            df_blur = df_blur[df_blur[score_col] != "N/A"].copy()
            df_blur[score_col] = pd.to_numeric(df_blur[score_col], errors="coerce")
            df_blur = df_blur[df_blur[score_col].notna()]
        
        if not df_blur.empty:
            plt.figure(figsize=(10, 6))
            sns.boxplot(data=df_blur, x='status', y=score_col)
            plt.title(f"{score_col}: Accepted vs Rejected")
            plt.yscale('log')
            plt.tight_layout()
            plt.savefig(self.output_dir / "focus_bias_analysis.png")
            # Keep legacy artifact name for older tests/reports.
            plt.savefig(self.output_dir / "blur_bias_analysis.png")
            plt.close()

        # 3. Confidence vs Status
        df_conf = df[df['yolo_confidence'] != "N/A"].copy()
        df_conf['yolo_confidence'] = df_conf['yolo_confidence'].astype(float)
        
        if not df_conf.empty:
            plt.figure(figsize=(10, 6))
            sns.violinplot(data=df_conf, x='status', y='yolo_confidence')
            plt.title("YOLO Confidence Distribution")
            plt.tight_layout()
            plt.savefig(self.output_dir / "confidence_bias_analysis.png")
            plt.close()

        disparity = self.compute_source_disparity(df)
        (self.output_dir / "filtering_bias_disparity.json").write_text(
            json.dumps(disparity, indent=2),
            encoding="utf-8",
        )

        logger.info(f"Bias analysis reports saved to {self.output_dir}")
