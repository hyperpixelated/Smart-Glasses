"""Load the project-wide traffic-sign taxonomy from the YOLO dataset config."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATASET_CONFIG = PROJECT_ROOT / "data" / "traffic_sign.yaml"


def load_dataset_config(config_path: Path = DEFAULT_DATASET_CONFIG) -> dict[str, Any]:
    """Load the central Ultralytics-compatible dataset configuration."""
    try:
        with config_path.open(encoding="utf-8") as config_file:
            config = yaml.safe_load(config_file)
    except OSError as error:
        raise ValueError(f"Unable to read dataset configuration: {config_path}") from error

    if not isinstance(config, dict):
        raise ValueError(f"Dataset configuration must be a YAML mapping: {config_path}")
    return config


def load_class_names(config_path: Path = DEFAULT_DATASET_CONFIG) -> tuple[str, ...]:
    """Return ordered class names from the single source-of-truth YAML config."""
    config = load_dataset_config(config_path)
    names = config.get("names")
    class_count = config.get("nc")

    if not isinstance(class_count, int) or class_count < 1:
        raise ValueError("Dataset configuration must define a positive integer 'nc'.")

    if isinstance(names, list):
        ordered_names = names
    elif isinstance(names, dict):
        try:
            ordered_names = [names[index] if index in names else names[str(index)] for index in range(class_count)]
        except KeyError as error:
            raise ValueError("Dataset configuration 'names' must define every class ID from 0 to nc - 1.") from error
    else:
        raise ValueError("Dataset configuration must define 'names' as a list or mapping.")

    if len(ordered_names) != class_count or not all(isinstance(name, str) and name for name in ordered_names):
        raise ValueError("Dataset configuration 'names' must contain one non-empty name per class.")

    return tuple(ordered_names)
