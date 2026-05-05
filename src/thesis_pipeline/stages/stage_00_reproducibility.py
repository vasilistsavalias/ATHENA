# src/thesis_pipeline/pipeline/stage_00_reproducibility.py
import logging
import json
import subprocess
import sys
import torch
import platform
from pathlib import Path
from thesis_pipeline.config_manager import ConfigManager

class ReproducibilityStage:
    def __init__(self, config_manager: ConfigManager):
        self.config_manager = config_manager
        self.output_dir = self.config_manager.get_stage_artifact_dir("S00")
        self.logger = logging.getLogger(__name__)

    def _write_environment_artifacts(self) -> None:
        env_file = self.output_dir / "environment.yml"
        requirements_file = self.output_dir / "requirements_freeze.txt"
        status = {
            "environment_yml_created": False,
            "requirements_freeze_created": False,
            "export_method": None,
        }

        try:
            with open(env_file, "w", encoding="utf-8") as handle:
                subprocess.check_call("conda env export", shell=True, stdout=handle)
            if env_file.stat().st_size <= 0:
                raise RuntimeError("conda env export produced an empty environment.yml")
            status["environment_yml_created"] = True
            status["export_method"] = "conda_env_export"
            self.logger.info("Exported conda environment.")
        except Exception:
            if env_file.exists() and env_file.stat().st_size == 0:
                env_file.unlink()
            self.logger.info("Conda export failed; falling back to pip freeze.")
            with open(requirements_file, "w", encoding="utf-8") as handle:
                subprocess.check_call([sys.executable, "-m", "pip", "freeze"], stdout=handle)
            status["requirements_freeze_created"] = True
            status["export_method"] = "pip_freeze"

        with open(self.output_dir / "environment_export_status.json", "w", encoding="utf-8") as handle:
            json.dump(status, handle, indent=4)

    def _write_git_state(self) -> None:
        project_root = Path(__file__).resolve().parents[3]

        def _git(*args: str) -> str | None:
            try:
                result = subprocess.run(
                    ["git", "-C", str(project_root), *args],
                    check=False,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                if result.returncode != 0:
                    return None
                return result.stdout.strip()
            except Exception:
                return None

        dirty_raw = _git("status", "--porcelain", "--untracked-files=no") or ""
        dirty_files = [line.strip() for line in dirty_raw.splitlines() if line.strip()]
        payload = {
            "commit": _git("rev-parse", "HEAD"),
            "short_commit": _git("rev-parse", "--short", "HEAD"),
            "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": bool(dirty_files),
            "dirty_files": dirty_files,
        }
        with open(self.output_dir / "git_state.json", "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=4)

    def run(self):
        self.logger.info("="*20 + " STAGE 00: Reproducibility Setup " + "="*20)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        try:
            self._write_environment_artifacts()
            self._write_git_state()

            # 2. Seeds
            seeds = {
                "global_random_state": self.config_manager.config.global_params.random_state,
                "numpy": 42,
                "torch": 42,
                "split_seed": 42
            }
            with open(self.output_dir / "seeds.json", "w", encoding="utf-8") as f:
                json.dump(seeds, f, indent=4)
            self.logger.info("Saved seeds registry.")

            # 3. Compute Log
            compute_info = {
                "platform": platform.platform(),
                "python_version": sys.version,
                "torch_version": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "cuda_version": torch.version.cuda if torch.cuda.is_available() else "N/A",
                "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
                "gpu_count": torch.cuda.device_count()
            }
            import pandas as pd
            pd.DataFrame([compute_info]).to_csv(self.output_dir / "compute_log.csv", index=False)
            self.logger.info("Logged compute hardware.")

            self.logger.info("="*20 + " STAGE 00 COMPLETED " + "="*20 + "\n")

        except Exception as e:
            self.logger.exception(f"Error in Reproducibility Stage: {e}")
            raise

if __name__ == '__main__':
    cm = ConfigManager()
    stage = ReproducibilityStage(cm)
    stage.run()
