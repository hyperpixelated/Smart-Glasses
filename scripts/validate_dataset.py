#!/usr/bin/env python3
"""Validate the local YOLO traffic-sign dataset without training or downloading data."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from smart_glasses.dataset.taxonomy import DEFAULT_DATASET_CONFIG, load_class_names, load_dataset_config
from smart_glasses.dataset.validation import validate_dataset


def main() -> int:
    """Validate processed images and labels using the central taxonomy."""
    try:
        config = load_dataset_config(DEFAULT_DATASET_CONFIG)
        class_names = load_class_names(DEFAULT_DATASET_CONFIG)
        dataset_root = (DEFAULT_DATASET_CONFIG.parent / config.get("path", ".")).resolve()
    except (KeyError, ValueError) as error:
        print(f"ERROR dataset_config: {error}")
        return 1

    report = validate_dataset(dataset_root, len(class_names))
    print("Smart Glasses Dataset Validation")
    print("=" * 32)
    print(f"Dataset root: {dataset_root}")
    print(f"Classes: {len(class_names)}")
    print(f"Images: {report.image_count}; labels: {report.label_count}")

    for issue in report.issues:
        print(f"{issue.level} {issue.code}: {issue.message}")

    if report.has_errors:
        print("ERROR Dataset validation failed.")
        return 1

    print("PASS Dataset validation completed without errors.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
