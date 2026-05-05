# src/thesis_pipeline/pipeline/stage_06_exploratory_data_analysis.py
import logging
from pathlib import Path
from thesis_pipeline.config_manager import ConfigManager
from thesis_pipeline.components.exploratory_data_analysis import ExploratoryDataAnalyzer

class ExploratoryDataAnalysisStage:
    def __init__(self, config_manager: ConfigManager):
        self.config_manager = config_manager
        self.config = config_manager.get_exploratory_data_analysis_config()
        self.paths = config_manager.get_paths()
        self.logger = logging.getLogger(__name__)

    def run(self):
        self.logger.info("="*20 + " STAGE 05: Exploratory Data Analysis " + "="*20)
        try:
            # BUG-FIX: Read from accepted/ subdir (Stage 03 output),
            # not the filtered root which also contains Stage 02b copies
            # and Stage 03 rejected images β€” rglob previously triple-counted.
            input_dir = Path(self.paths.data.filtered) / "accepted"
            output_dir = self.config_manager.get_stage_artifact_dir("S06")
            output_dir.mkdir(parents=True, exist_ok=True)

            analyzer = ExploratoryDataAnalyzer(
                input_dir=input_dir,
                output_dir=output_dir,
                extensions=self.config.image_extensions
            )
            analyzer.run()
            self.logger.info("="*20 + " STAGE 05 COMPLETED " + "="*20 + "\n")
        except Exception as e:
            self.logger.exception(f"Error in EDA stage: {e}")
            raise

if __name__ == '__main__':
    cm = ConfigManager()
    stage = ExploratoryDataAnalysisStage(cm)
    stage.run()

