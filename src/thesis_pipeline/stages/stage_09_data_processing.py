# src/thesis_pipeline/pipeline/stage_09_data_processing.py
import logging
import json
from pathlib import Path
from thesis_pipeline.config_manager import ConfigManager
from thesis_pipeline.components.processing import ImageProcessor
from thesis_pipeline.utils.common import save_json

class DataProcessingStage:
    def __init__(self, config_manager: ConfigManager):
        self.config_manager = config_manager
        self.config = config_manager.get_data_processing_config()
        self.global_params = config_manager.get_global_params()
        self.paths = config_manager.get_paths()
        self.logger = logging.getLogger(__name__)

    def _resolve_dataset_limit(self):
        try:
            global_limit = self.global_params.get("dataset_limit")
        except Exception:
            global_limit = None

        try:
            stage_limit = self.config.get("limit")
        except Exception:
            stage_limit = None

        limits = [v for v in (global_limit, stage_limit) if isinstance(v, int) and v > 0]
        return min(limits) if limits else None

    def run(self):
        self.logger.info("="*20 + " STAGE 09: Data Processing " + "="*20)
        try:
            input_dir = Path(self.paths.data.filtered) / "accepted"
            output_dir = Path(self.paths.data.processed)
            report_dir = self.config_manager.get_stage_artifact_dir("S09")
            report_dir.mkdir(parents=True, exist_ok=True)
            output_dir.mkdir(parents=True, exist_ok=True)

            processor = ImageProcessor(
                input_dir=input_dir,
                output_dir=output_dir,
                report_dir=report_dir,
                image_size=tuple(self.config.image_size),
                output_format=self.config.format
            )
            limit = self._resolve_dataset_limit()
            if limit:
                self.logger.info(f"Applying dataset limit: {limit} images.")
            summary = processor.process_images(max_images=limit)
            
            # Save summary to both data and artifacts
            summary_path = output_dir / "processing_summary.json"
            save_json(summary_path, summary)
            save_json(report_dir / "processing_summary.json", summary)
            
            self.logger.info(f"Processed {summary['processed_count']} images.")
            self.logger.info("="*20 + " STAGE 09 COMPLETED " + "="*20 + "\n")
        except Exception as e:
            self.logger.exception(f"Error in Processing stage: {e}")
            raise

if __name__ == '__main__':
    cm = ConfigManager()
    stage = DataProcessingStage(cm)
    stage.run()

