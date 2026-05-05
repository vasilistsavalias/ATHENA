# src/thesis_pipeline/pipeline/stage_10_data_splitting.py
import logging
import json
from pathlib import Path
from thesis_pipeline.config_manager import ConfigManager
from thesis_pipeline.components.splitting import DataSplitter

class DataSplittingStage:
    def __init__(self, config_manager: ConfigManager):
        self.config_manager = config_manager
        self.config = config_manager.get_data_splitting_config()
        self.paths = config_manager.get_paths()
        self.global_params = config_manager.get_global_params()
        self.logger = logging.getLogger(__name__)

    def run(self):
        self.logger.info("="*20 + " STAGE 10: Data Splitting " + "="*20)
        try:
            input_dir = Path(self.paths.data.processed)
            output_dir = Path(self.paths.data.splits)
            report_dir = self.config_manager.get_stage_artifact_dir("S10")
            report_dir.mkdir(parents=True, exist_ok=True)
            output_dir.mkdir(parents=True, exist_ok=True)

            # Extract ratios from config
            # config.ratios is expected to be [train, val, test]
            val_size = self.config.ratios[1]
            test_size = self.config.ratios[2]

            splitter = DataSplitter(
                input_dir=input_dir,
                output_dir=output_dir,
                report_dir=report_dir,
                test_size=test_size,
                validation_size=val_size,
                random_state=self.global_params.random_state
            )
            splitter.split_data()
            
            split_manifests = {}
            stats = {}
            for split in ("train", "validation", "test"):
                files = sorted(
                    p.name for p in (output_dir / split).iterdir()
                    if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg"}
                ) if (output_dir / split).exists() else []
                split_manifests[split] = files
                stats[split] = len(files)
            stats["total"] = sum(stats.values())
            with open(report_dir / "split_stats.json", "w") as f: json.dump(stats, f, indent=4)
            with open(report_dir / "split_manifest.json", "w", encoding="utf-8") as f:
                json.dump(split_manifests, f, indent=4)
            self.logger.info("="*20 + " STAGE 10 COMPLETED " + "="*20 + "\n")
        except Exception as e:
            self.logger.exception(f"Error in Splitting stage: {e}")
            raise

if __name__ == '__main__':
    cm = ConfigManager()
    stage = DataSplittingStage(cm)
    stage.run()

