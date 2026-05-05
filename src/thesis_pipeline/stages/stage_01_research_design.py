# src/thesis_pipeline/pipeline/stage_01_research_design.py
import logging
import shutil
from pathlib import Path
from thesis_pipeline.config_manager import ConfigManager

class ResearchDesignStage:
    def __init__(self, config_manager: ConfigManager):
        self.config_manager = config_manager
        self.output_dir = self.config_manager.get_stage_artifact_dir("S01")
        # Assuming docs are in project_root/docs/research_design
        # config_manager doesn't track docs path explicitly usually, so we resolve relative to data path or root
        self.docs_source = Path("docs/research_design")
        self.logger = logging.getLogger(__name__)

    def run(self):
        self.logger.info("="*20 + " STAGE 01: Research Design Documentation " + "="*20)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        required_files = [
            "hypotheses.json",
            "success_criteria.md",
            "research_questions.md",
            "novelty_claim.md",
            "literature_gap_analysis.md"
        ]

        missing = []
        for filename in required_files:
            src = self.docs_source / filename
            dst = self.output_dir / filename
            
            if src.exists():
                shutil.copy(src, dst)
                self.logger.info(f"Verified & Archived: {filename}")
            else:
                missing.append(filename)
                self.logger.warning(f"MISSING DOCUMENT: {filename}")

        if missing:
            self.logger.error(f"Critical Research Design Documents Missing: {missing}")
            # We don't raise exception to avoid blocking pipeline, but this is a Thesis FAIL condition.
            with open(self.output_dir / "MISSING_DOCS_REPORT.txt", "w") as f:
                f.write(f"The following mandatory documents were missing at run time: {missing}\n")
        else:
            self.logger.info("All research design documents present.")

        self.logger.info("="*20 + " STAGE 01 COMPLETED " + "="*20 + "\n")

if __name__ == '__main__':
    cm = ConfigManager()
    stage = ResearchDesignStage(cm)
    stage.run()
