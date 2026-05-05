import shutil
import datetime
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def archive_outputs():
    """
    Archives specific subdirectories within the 'outputs' directory to a new timestamped archive folder.
    """
    project_root = Path(__file__).parent.parent.parent
    outputs_dir = project_root / "outputs"
    
    if not outputs_dir.exists():
        logger.warning(f"Outputs directory not found: {outputs_dir}")
        return

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_name = f"archive_{timestamp}"
    archive_path = outputs_dir / archive_name
    
    # Define directories to archive (excluding existing archives and smoke_test if desired, 
    # but usually we want to archive the main artifacts)
    # Based on file structure, we have numbered folders like '02_intelligent_filtering', etc.
    # We should archive everything that isn't an archive itself.
    
    items_to_archive = []
    for item in outputs_dir.iterdir():
        if item.is_dir() and not item.name.startswith("archive_") and item.name != "smoke_test":
             items_to_archive.append(item)
    
    if not items_to_archive:
        logger.info("No items found to archive.")
        return

    logger.info(f"Creating archive at: {archive_path}")
    archive_path.mkdir(parents=True, exist_ok=True)

    for item in items_to_archive:
        destination = archive_path / item.name
        try:
            logger.info(f"Moving {item.name}...")
            shutil.move(str(item), str(destination))
        except Exception as e:
            logger.error(f"Failed to move {item.name}: {e}")

    logger.info("Archival complete.")

if __name__ == "__main__":
    archive_outputs()
