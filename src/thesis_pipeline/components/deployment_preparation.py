# src/thesis_pipeline/components/deployment_preparation.py
import json
import logging
import shutil
from pathlib import Path

class DeploymentPackager:
    def __init__(self, config, model_input_dir: Path, hyperparams_input_file: Path, output_dir: Path = None):
        self.config = config
        self.model_input_dir = model_input_dir
        self.hyperparams_input_file = hyperparams_input_file
        # Accept output_dir explicitly; fall back to config only if available.
        if output_dir is not None:
            self.output_dir = Path(output_dir)
        elif hasattr(config, 'output_dir'):
            self.output_dir = Path(config.output_dir)
        else:
            raise ValueError(
                "output_dir must be passed explicitly or present in config.output_dir"
            )
        self.logger = logging.getLogger(__name__)

    def package(self):
        """Packages the model and necessary files for deployment."""
        if self.output_dir.exists():
            self.logger.warning(f"Output directory {self.output_dir} exists. Cleaning it.")
            shutil.rmtree(self.output_dir)
        self.output_dir.mkdir(parents=True)
        self.logger.info(f"Created deployment package directory at: {self.output_dir}")

        # Copy UNet model
        unet_dest_dir = self.output_dir / 'unet_final'
        shutil.copytree(self.model_input_dir, unet_dest_dir)
        self.logger.info(f"Copied UNet to: {unet_dest_dir}")
        weight_candidates = (
            list(unet_dest_dir.glob("*.safetensors"))
            + list(unet_dest_dir.glob("*.bin"))
            + list(unet_dest_dir.glob("*.pt"))
            + list(unet_dest_dir.glob("*.pth"))
        )
        if not weight_candidates:
            raise RuntimeError(
                f"Deployment package missing model weights in {unet_dest_dir}. "
                "Expected at least one *.safetensors/*.bin/*.pt/*.pth file."
            )

        # Copy hyperparameters
        shutil.copy(self.hyperparams_input_file, self.output_dir)
        self.logger.info(f"Copied hyperparameters to: {self.output_dir}")

        # Create README
        readme_content = f"""
# Inpainting Model Deployment Package
- `/unet_final`: The fine-tuned UNet model weights.
- `{self.hyperparams_input_file.name}`: The hyperparameters.
"""
        with open(self.output_dir / "README.md", 'w') as f:
            f.write(readme_content)
        self.logger.info("Created README.md for the package.")
        (self.output_dir / "package_summary.json").write_text(
            json.dumps(
                {
                    "model_input_dir": str(self.model_input_dir),
                    "output_dir": str(self.output_dir),
                    "weights": [p.name for p in sorted(weight_candidates)],
                    "hyperparameters_file": self.hyperparams_input_file.name,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
