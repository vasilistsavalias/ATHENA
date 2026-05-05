import sys
from pathlib import Path
from loguru import logger

def setup_logger(stage_name: str, log_file_path: Path):
    """
    Configures the Loguru logger for a specific pipeline stage.

    This function removes any existing handlers and sets up two new ones:
    1. A console sink for clear, real-time status updates (INFO level).
    2. A file sink for detailed, debug-level logging for forensic analysis.

    Args:
        stage_name (str): The name of the stage (e.g., "01_Data_Acquisition").
        log_file_path (Path): The full path to the log file for the stage.
    """
    logger.remove()

    # Console sink for human-readable output
    logger.add(
        sys.stderr,
        level="INFO",
        format=(
            f"<white>{{time:HH:mm:ss}}</white> | "
            f"<bold><blue>{stage_name}</blue></bold> | "
            "<level>{level: <8}</level> | "
            "<level>{message}</level>"
        ),
        colorize=True
    )

    # File sink for detailed, persistent logs
    log_file_path.parent.mkdir(parents=True, exist_ok=True)
    logger.add(
        log_file_path,
        level="DEBUG",
        format=(
            "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
            "{level: <8} | "
            "{name}:{function}:{line} - {message}"
        ),
        rotation="10 MB",
        retention="10 days"
    )

    return logger
