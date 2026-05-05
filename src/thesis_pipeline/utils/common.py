# src/thesis_pipeline/utils/common.py
import json
import yaml
import pickle
import logging
from pathlib import Path
from box import ConfigBox

logger = logging.getLogger(__name__)

def save_json(path: Path, data: dict):
    """Saves a dictionary to a JSON file."""
    try:
        with open(path, "w") as f:
            json.dump(data, f, indent=4)
        logger.info(f"JSON file saved successfully at: {path}")
    except Exception as e:
        logger.error(f"Error saving JSON file at {path}: {e}")
        raise

def load_json(path: Path) -> ConfigBox:
    """Loads a JSON file and returns its content as a ConfigBox."""
    try:
        with open(path, "r") as f:
            content = json.load(f)
        logger.info(f"JSON file loaded successfully from: {path}")
        return ConfigBox(content)
    except Exception as e:
        logger.error(f"Error loading JSON file from {path}: {e}")
        raise

def save_yaml(path: Path, data: dict):
    """Saves a dictionary to a YAML file."""
    try:
        with open(path, "w") as f:
            yaml.dump(data, f)
        logger.info(f"YAML file saved successfully at: {path}")
    except Exception as e:
        logger.error(f"Error saving YAML file at {path}: {e}")
        raise

def load_yaml(path: Path) -> ConfigBox:
    """Loads a YAML file and returns its content as a ConfigBox."""
    try:
        with open(path, "r") as f:
            content = yaml.safe_load(f)
        logger.info(f"YAML file loaded successfully from: {path}")
        return ConfigBox(content)
    except Exception as e:
        logger.error(f"Error loading YAML file from {path}: {e}")
        raise

def save_binary(path: Path, data: any):
    """Saves data as a binary file (e.g., a trained model)."""
    try:
        with open(path, "wb") as f:
            pickle.dump(data, f)
        logger.info(f"Binary file saved successfully at: {path}")
    except Exception as e:
        logger.error(f"Error saving binary file at {path}: {e}")
        raise

def load_binary(path: Path) -> any:
    """Loads data from a binary file."""
    try:
        with open(path, "rb") as f:
            data = pickle.load(f)
        logger.info(f"Binary file loaded successfully from: {path}")
        return data
    except Exception as e:
        logger.error(f"Error loading binary file from {path}: {e}")
        raise

def get_file_size(path: Path) -> str:
    """Returns the size of a file in kilobytes (KB)."""
    try:
        size_in_bytes = path.stat().st_size
        size_in_kb = round(size_in_bytes / 1024)
        return f"~ {size_in_kb} KB"
    except FileNotFoundError:
        return "File not found"
    except Exception as e:
        logger.error(f"Error getting file size for {path}: {e}")
        return "Size unavailable"
