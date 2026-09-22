"""Tests for read-only DataCluster Pascal VOC inspection helpers."""

import struct
from pathlib import Path

from scripts.inspect_datacluster_dataset import inspect_dataset, is_valid_bbox, parse_voc_annotation


def _write_image(root: Path, name: str = "sample.bmp") -> Path:
    image_path = root / name
    # A valid 1x1, 24-bit BMP assembled with only the standard library.
    file_header = b"BM" + struct.pack("<IHHI", 58, 0, 0, 54)
    dib_header = struct.pack("<IIIHHIIIIII", 40, 1, 1, 1, 24, 0, 4, 0, 0, 0, 0)
    image_path.write_bytes(file_header + dib_header + b"\x00\x00\x00\x00")
    return image_path


def _write_xml(root: Path, filename: str, box: tuple[float, float, float, float]) -> Path:
    xmin, ymin, xmax, ymax = box
    xml_path = root / f"{Path(filename).stem}.xml"
    xml_path.write_text(
        f"""<annotation>
<filename>{filename}</filename>
<object><name>traffic_sign</name><bndbox>
<xmin>{xmin}</xmin><ymin>{ymin}</ymin><xmax>{xmax}</xmax><ymax>{ymax}</ymax>
</bndbox></object>
</annotation>""",
        encoding="utf-8",
    )
    return xml_path


def test_parser_reads_voc_object_and_coordinates(tmp_path: Path) -> None:
    xml_path = _write_xml(tmp_path, "sample.bmp", (0, 0, 1, 1))

    annotation = parse_voc_annotation(xml_path)

    assert annotation.declared_filename == "sample.bmp"
    assert annotation.objects == (("traffic_sign", 0.0, 0.0, 1.0, 1.0),)


def test_inspection_counts_valid_synthetic_pair(tmp_path: Path) -> None:
    _write_image(tmp_path)
    _write_xml(tmp_path, "sample.bmp", (0, 0, 1, 1))

    report = inspect_dataset(tmp_path)

    assert report.matched_pair_count == 1
    assert len(report.valid_boxes) == 1
    assert not report.invalid_boxes
    assert report.class_distribution == {"traffic_sign": 1}


def test_inspection_marks_out_of_bounds_box_invalid(tmp_path: Path) -> None:
    _write_image(tmp_path)
    _write_xml(tmp_path, "sample.bmp", (0, 0, 2, 1))

    report = inspect_dataset(tmp_path)

    assert len(report.invalid_boxes) == 1
    assert not is_valid_bbox(report.invalid_boxes[0])


def test_inspection_reports_missing_counterparts(tmp_path: Path) -> None:
    _write_image(tmp_path, "image_only.png")
    _write_xml(tmp_path, "xml_only.png", (0, 0, 1, 1))

    report = inspect_dataset(tmp_path)

    assert len(report.missing_annotations) == 1
    assert len(report.missing_images) == 1
