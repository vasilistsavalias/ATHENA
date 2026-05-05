# src/thesis_pipeline/pipeline/stage_16_deployment_preparation.py
import logging
import shutil
from pathlib import Path
from thesis_pipeline.config_manager import ConfigManager
from thesis_pipeline.components.deployment_preparation import DeploymentPackager

class DeploymentPreparationStage:
    def __init__(self, config_manager: ConfigManager):
        self.config_manager = config_manager
        self.paths = config_manager.get_paths()
        self.logger = logging.getLogger(__name__)
        self.output_dir = self.config_manager.get_stage_artifact_dir("S16")

    def run(self):
        self.logger.info("="*20 + " STAGE 16: Deployment Preparation " + "="*20)
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            models_dir = Path(self.paths.data.models)
            best = models_dir / "unet_best"
            final = models_dir / "unet_final"
            model_input = best if best.exists() else final
            hyperparams_input = self.config_manager.get_stage_artifact_path("S12", "best_hyperparameters.yaml")
            
            if not model_input.exists():
                self.logger.warning("No model found to package. Skipping.")
                return
            if not hyperparams_input.exists():
                raise FileNotFoundError(f"Missing hyperparameters file: {hyperparams_input}")

            packager = DeploymentPackager(
                self.config_manager.config.deployment_preparation,
                model_input, hyperparams_input,
                output_dir=self.output_dir,
            )
            packager.package()
            
            self.logger.info("="*20 + " STAGE 16 COMPLETED " + "="*20 + "\n")
        except Exception as e:
            self.logger.exception(f"Error: {e}")

if __name__ == '__main__':
    cm = ConfigManager(); stage = DeploymentPreparationStage(cm); stage.run()

