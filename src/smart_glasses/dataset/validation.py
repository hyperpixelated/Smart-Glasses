"""Validation logic for an Ultralytics YOLO detection dataset."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class ValidationIssue:
    """One dataset validation finding."""

    level: str
    code: str
    message: str


@dataclass
class ValidationReport:
    """Collected findings and counts from one dataset validation pass."""

    issues: list[ValidationIssue] = field(default_factory=list)
    image_count: int = 0
    label_count: int = 0

    @property
    def has_errors(self) -> bool:
        """Whether the report contains an error that should fail validation."""
        return any(issue.level == "ERROR" for issue in self.issues)

    def add(self, level: str, code: str, message: str) -> None:
        """Add a structured validation finding."""
        self.issues.append(ValidationIssue(level, code, message))


def _image_files(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(path for path in directory.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)


def _label_files(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(directory.rglob("*.txt"))


def _expected_label_path(image_path: Path, images_dir: Path, labels_dir: Path) -> Path:
    return (labels_dir / image_path.relative_to(images_dir)).with_suffix(".txt")


def _has_matching_image(label_path: Path, images_dir: Path, labels_dir: Path) -> bool:
    relative_stem = label_path.relative_to(labels_dir).with_suffix("")
    return any((images_dir / relative_stem).with_suffix(suffix).is_file() for suffix in IMAGE_SUFFIXES)


def _validate_annotation_file(label_path: Path, class_count: int, report: ValidationReport) -> None:
    try:
        lines = label_path.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError:
        report.add("ERROR", "unreadable_label", f"{label_path}: label file is not UTF-8 text.")
        return

    if not lines:
        report.add("WARNING", "empty_annotation", f"{label_path}: annotation file is empty.")
        return

    for line_number, line in enumerate(lines, start=1):
        parts = line.split()
        location = f"{label_path}:{line_number}"
        if len(parts) != 5:
            report.add("ERROR", "malformed_annotation", f"{location}: expected 5 whitespace-separated values.")
            continue

        class_token, *coordinate_tokens = parts
        try:
            class_id = int(class_token)
        except ValueError:
            report.add("ERROR", "invalid_class_id", f"{location}: class ID must be an integer.")
            continue

        if class_token != str(class_id) or not 0 <= class_id < class_count:
            report.add("ERROR", "invalid_class_id", f"{location}: class ID {class_token!r} is outside 0..{class_count - 1}.")
            continue

        try:
            x_center, y_center, width, height = (float(value) for value in coordinate_tokens)
        except ValueError:
            report.add("ERROR", "malformed_annotation", f"{location}: coordinates must be numeric.")
            continue

        if not all(0.0 <= value <= 1.0 for value in (x_center, y_center, width, height)):
            report.add("ERROR", "coordinates_out_of_range", f"{location}: coordinates must be within [0, 1].")
        if width <= 0.0 or height <= 0.0:
            report.add("ERROR", "invalid_bbox_size", f"{location}: bounding-box width and height must be greater than 0.")


def validate_dataset(root: Path, class_count: int) -> ValidationReport:
    """Validate processed YOLO images and labels under *root* for all standard splits."""
    report = ValidationReport()
    filenames_by_split: dict[str, set[str]] = {}

    for split in SPLITS:
        images_dir = root / "images" / split
        labels_dir = root / "labels" / split
        images = _image_files(images_dir)
        labels = _label_files(labels_dir)
        report.image_count += len(images)
        report.label_count += len(labels)
        filenames_by_split[split] = {image.name for image in images}

        for image_path in images:
            expected_label = _expected_label_path(image_path, images_dir, labels_dir)
            if not expected_label.is_file():
                report.add("ERROR", "missing_label", f"{image_path}: missing label file {expected_label}.")

        for label_path in labels:
            if not _has_matching_image(label_path, images_dir, labels_dir):
                report.add("ERROR", "missing_image", f"{label_path}: no corresponding image file exists.")
            _validate_annotation_file(label_path, class_count, report)

    if report.image_count == 0 and report.label_count == 0:
        report.add("WARNING", "empty_dataset", "No images or labels exist yet; dataset validation is otherwise complete.")

    filename_splits: dict[str, list[str]] = defaultdict(list)
    for split, filenames in filenames_by_split.items():
        for filename in filenames:
            filename_splits[filename].append(split)

    for filename, splits in sorted(filename_splits.items()):
        if len(splits) > 1:
            split_list = ", ".join(splits)
            report.add("ERROR", "duplicate_filename_across_splits", f"{filename}: duplicate image filename in {split_list}.")
            report.add("ERROR", "split_leakage_identical_filename", f"{filename}: identical filename indicates train/val/test leakage across {split_list}.")

    return report
