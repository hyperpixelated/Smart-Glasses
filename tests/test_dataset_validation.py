"""Tests for YOLO dataset validation using temporary local files."""

from pathlib import Path

from smart_glasses.dataset.validation import validate_dataset


CLASS_COUNT = 10


def _create_image(root: Path, split: str, name: str = "sample.jpg") -> Path:
    image_path = root / "images" / split / name
    image_path.parent.mkdir(parents=True, exist_ok=True)
    image_path.write_bytes(b"placeholder")
    return image_path


def _create_label(root: Path, split: str, contents: str, name: str = "sample.txt") -> Path:
    label_path = root / "labels" / split / name
    label_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.write_text(contents, encoding="utf-8")
    return label_path


def _codes(root: Path) -> set[str]:
    return {issue.code for issue in validate_dataset(root, CLASS_COUNT).issues}


def test_valid_annotation_has_no_errors(tmp_path: Path) -> None:
    _create_image(tmp_path, "train")
    _create_label(tmp_path, "train", "0 0.5 0.5 0.2 0.2\n")

    report = validate_dataset(tmp_path, CLASS_COUNT)

    assert not report.has_errors


def test_invalid_class_id_is_reported(tmp_path: Path) -> None:
    _create_image(tmp_path, "train")
    _create_label(tmp_path, "train", "10 0.5 0.5 0.2 0.2\n")

    assert "invalid_class_id" in _codes(tmp_path)


def test_coordinates_outside_unit_range_are_reported(tmp_path: Path) -> None:
    _create_image(tmp_path, "train")
    _create_label(tmp_path, "train", "0 1.1 0.5 0.2 0.2\n")

    assert "coordinates_out_of_range" in _codes(tmp_path)


def test_malformed_annotation_is_reported(tmp_path: Path) -> None:
    _create_image(tmp_path, "train")
    _create_label(tmp_path, "train", "0 0.5 0.5 0.2\n")

    assert "malformed_annotation" in _codes(tmp_path)


def test_missing_label_is_reported(tmp_path: Path) -> None:
    _create_image(tmp_path, "train")

    assert "missing_label" in _codes(tmp_path)


def test_missing_image_is_reported(tmp_path: Path) -> None:
    _create_label(tmp_path, "train", "0 0.5 0.5 0.2 0.2\n")

    assert "missing_image" in _codes(tmp_path)
