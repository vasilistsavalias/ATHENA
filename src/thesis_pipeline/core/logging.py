"""Canonical logging setup for ATHENA."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from loguru import logger

from thesis_pipeline.core.config import ConfigManager


class LoggingConfig:
    """Configure Loguru plus stdlib logging interception."""

    def __init__(self, config_manager: ConfigManager):
        self.config = config_manager.get_logging_config()
        self.log_dir = Path(self.config.log_dir)
        self.log_file = self.log_dir / self.config.main_log_file
        self.log_level = self.config.log_level.upper()

    def setup_logging(self):
        self.log_dir.mkdir(parents=True, exist_ok=True)
        logger.remove()

        logger.add(
            sys.stdout,
            level=self.log_level,
            format=(
                "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
                "<level>{level: <8}</level> | "
                "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
                "<level>{message}</level>"
            ),
        )
        logger.add(
            self.log_file,
            level=self.log_level,
            rotation="10 MB",
            retention="7 days",
            enqueue=True,
            backtrace=True,
            diagnose=True,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        )

        class InterceptHandler(logging.Handler):
            def emit(self, record):
                try:
                    level = logger.level(record.levelname).name
                except ValueError:
                    level = record.levelno

                frame, depth = logging.currentframe(), 2
                while frame and frame.f_code.co_filename == logging.__file__:
                    frame = frame.f_back
                    depth += 1

                logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())

        logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)
        logging.getLogger(__name__).info(
            "Logging configured successfully. All logs will be sent to console and file."
        )
