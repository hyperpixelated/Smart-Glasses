#!/usr/bin/env python3
"""Read-only inspection of Pascal VOC annotations in the DataCluster dataset."""

from __future__ import annotations

import statistics
import sys
import xml.etree.ElementTree as element_tree
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import cv2


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT / "data" / "raw" / "datacluster"
IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


@dataclass(frozen=True)
class BoundingBox:
    """One Pascal VOC bounding box paired with its image metadata."""

    image_name: str
    class_name: str
    xmin: float
    ymin: float
    xmax: float
    ymax: float
    image_width: int
    image_height: int

    @property
    def width(self) -> float:
        return self.xmax - self.xmin

    @property
    def height(self) -> float:
        return self.ymax - self.ymin

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def area_percent(self) -> float:
        return 100.0 * self.area / (self.image_width * self.image_height)


@dataclass(frozen=True)
class ParsedAnnotation:
    """The filename declared by one XML file and its parsed objects."""

    declared_filename: str
    objects: tuple[tuple[str, float, float, float, float], ...]


@dataclass
class InspectionReport:
    """Collected read-only findings and measurements for the dataset."""

    image_count: int = 0
    xml_count: int = 0
    matched_pair_count: int = 0
    missing_images: list[Path] = field(default_factory=list)
    missing_annotations: list[Path] = field(default_factory=list)
    invalid_xml_filenames: list[Path] = field(default_factory=list)
    invalid_boxes: list[BoundingBox] = field(default_factory=list)
    valid_boxes: list[BoundingBox] = field(default_factory=list)
    class_distribution: Counter[str] = field(default_factory=Counter)
    dimension_distribution: Counter[tuple[int, int]] = field(default_factory=Counter)
    duplicate_filenames: dict[str, list[Path]] = field(default_factory=dict)
    parse_errors: list[str] = field(default_factory=list)
    unreadable_images: list[Path] = field(default_factory=list)


def parse_voc_annotation(xml_path: Path) -> ParsedAnnotation:
    """Parse one Pascal VOC XML file without changing the source file."""
    root = element_tree.parse(xml_path).getroot()
    declared_filename = (root.findtext("filename") or "").strip()
    objects = []

    for object_node in root.findall("object"):
        class_name = (object_node.findtext("name") or "").strip()
        box = object_node.find("bndbox")
        if not class_name or box is None:
            raise ValueError("object requires both a name and bndbox")
        try:
            coordinates = tuple(float(box.findtext(name, "")) for name in ("xmin", "ymin", "xmax", "ymax"))
        except ValueError as error:
            raise ValueError("bounding-box coordinates must be numeric") from error
        objects.append((class_name, *coordinates))

    return ParsedAnnotation(declared_filename, tuple(objects))


def is_valid_bbox(box: BoundingBox) -> bool:
    """Return whether box ordering and image-boundary constraints are satisfied."""
    return (
        box.xmin < box.xmax
        and box.ymin < box.ymax
        and 0 <= box.xmin <= box.image_width
        and 0 <= box.xmax <= box.image_width
        and 0 <= box.ymin <= box.image_height
        and 0 <= box.ymax <= box.image_height
    )


def _find_files(root: Path, suffixes: set[str]) -> list[Path]:
    return sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in suffixes)


def _median_range(values: list[float], precision: int = 2) -> str:
    if not values:
        return "n/a"
    return (
        f"min={min(values):.{precision}f}, max={max(values):.{precision}f}, "
        f"median={statistics.median(values):.{precision}f}"
    )


def inspect_dataset(dataset_root: Path = DATASET_ROOT) -> InspectionReport:
    """Inspect image/XML pairs recursively and return a report without writing files."""
    report = InspectionReport()
    images = _find_files(dataset_root, IMAGE_SUFFIXES)
    xml_files = _find_files(dataset_root, {".xml"})
    report.image_count = len(images)
    report.xml_count = len(xml_files)

    images_by_stem: dict[str, list[Path]] = defaultdict(list)
    images_by_name: dict[str, list[Path]] = defaultdict(list)
    for image_path in images:
        images_by_stem[image_path.stem].append(image_path)
        images_by_name[image_path.name].append(image_path)
    report.duplicate_filenames = {name: paths for name, paths in images_by_name.items() if len(paths) > 1}

    xml_by_stem: dict[str, list[Path]] = defaultdict(list)
    for xml_path in xml_files:
        xml_by_stem[xml_path.stem].append(xml_path)

    for image_path in images:
        if image_path.stem not in xml_by_stem:
            report.missing_annotations.append(image_path)

    for xml_path in xml_files:
        matching_images = images_by_stem.get(xml_path.stem, [])
        if not matching_images:
            report.missing_images.append(xml_path)
            continue
        if len(matching_images) != 1:
            report.parse_errors.append(f"{xml_path}: ambiguous image filename match.")
            continue

        image_path = matching_images[0]
        try:
            annotation = parse_voc_annotation(xml_path)
        except (OSError, element_tree.ParseError, ValueError) as error:
            report.parse_errors.append(f"{xml_path}: {error}")
            continue

        if annotation.declared_filename not in images_by_name:
            report.invalid_xml_filenames.append(xml_path)

        image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        if image is None:
            report.unreadable_images.append(image_path)
            continue
        image_height, image_width = image.shape[:2]
        report.dimension_distribution[(image_width, image_height)] += 1
        report.matched_pair_count += 1

        for class_name, xmin, ymin, xmax, ymax in annotation.objects:
            report.class_distribution[class_name] += 1
            box = BoundingBox(image_path.name, class_name, xmin, ymin, xmax, ymax, image_width, image_height)
            if is_valid_bbox(box):
                report.valid_boxes.append(box)
            else:
                report.invalid_boxes.append(box)

    return report


def print_report(report: InspectionReport) -> None:
    """Print a concise, human-readable summary of an inspection report."""
    widths = [box.width for box in report.valid_boxes]
    heights = [box.height for box in report.valid_boxes]
    area_percents = [box.area_percent for box in report.valid_boxes]

    print("DataCluster Traffic-Sign Dataset Inspection")
    print("=" * 43)
    print(f"Total images: {report.image_count}")
    print(f"Total XML annotations: {report.xml_count}")
    print(f"Matched pairs: {report.matched_pair_count}")
    print(f"Missing images: {len(report.missing_images)}")
    print(f"Missing annotations: {len(report.missing_annotations)}")
    print(f"Total objects: {sum(report.class_distribution.values())}")
    print(f"Invalid bounding boxes: {len(report.invalid_boxes)}")
    print(f"Duplicate image filenames: {len(report.duplicate_filenames)}")
    print(f"XML filenames without an existing image: {len(report.invalid_xml_filenames)}")

    print("\nClass distribution:")
    for class_name, count in sorted(report.class_distribution.items()):
        print(f"  {class_name}: {count}")

    print("\nImage dimension distribution:")
    for (width, height), count in sorted(report.dimension_distribution.items()):
        print(f"  {width}x{height}: {count}")

    print("\nBounding-box measurements for valid boxes:")
    print(f"  Width (pixels): {_median_range(widths)}")
    print(f"  Height (pixels): {_median_range(heights)}")
    print(f"  Area (% of image): {_median_range(area_percents, precision=4)}")

    print("\n10 smallest valid bounding boxes by image area:")
    print("  Rank  Image                                Class          Width  Height  Area %")
    for rank, box in enumerate(sorted(report.valid_boxes, key=lambda item: item.area_percent)[:10], start=1):
        print(
            f"  {rank:>4}  {box.image_name[:36]:<36}  {box.class_name:<13}  "
            f"{box.width:>5.1f}  {box.height:>6.1f}  {box.area_percent:>6.3f}"
        )

    if report.parse_errors or report.unreadable_images:
        print("\nERROR Inspection issues:")
        for error in report.parse_errors:
            print(f"  {error}")
        for image_path in report.unreadable_images:
            print(f"  {image_path}: unable to read image data.")


def main() -> int:
    """Run the read-only inspection against the checked-in raw dataset location."""
    if not DATASET_ROOT.is_dir():
        print(f"ERROR: DataCluster dataset directory not found: {DATASET_ROOT}", file=sys.stderr)
        return 1

    report = inspect_dataset(DATASET_ROOT)
    print_report(report)
    return 1 if report.parse_errors or report.unreadable_images else 0


if __name__ == "__main__":
    raise SystemExit(main())
