import os
import random
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

SOURCE = Path("data/raw/datacluster")
OUTPUT = Path("data/processed/datacluster_yolo")

SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.20

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def find_images():
    images = {}

    for path in SOURCE.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            images[path.name] = path

    return images


def find_annotations():
    annotations = {}

    for path in SOURCE.rglob("*.xml"):
        root = ET.parse(path).getroot()
        filename = root.findtext("filename")

        if filename:
            annotations[Path(filename).name] = path

    return annotations


def convert_bbox(obj, image_width, image_height):
    box = obj.find("bndbox")

    xmin = float(box.findtext("xmin"))
    ymin = float(box.findtext("ymin"))
    xmax = float(box.findtext("xmax"))
    ymax = float(box.findtext("ymax"))

    # Clamp to image boundaries.
    xmin = max(0.0, min(xmin, image_width))
    xmax = max(0.0, min(xmax, image_width))
    ymin = max(0.0, min(ymin, image_height))
    ymax = max(0.0, min(ymax, image_height))

    box_width = xmax - xmin
    box_height = ymax - ymin

    if box_width <= 0 or box_height <= 0:
        raise ValueError("Invalid bounding box")

    x_center = (xmin + xmax) / 2.0 / image_width
    y_center = (ymin + ymax) / 2.0 / image_height
    width = box_width / image_width
    height = box_height / image_height

    return f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"


def process_annotation(xml_path, image_path):
    root = ET.parse(xml_path).getroot()

    size = root.find("size")

    image_width = float(size.findtext("width"))
    image_height = float(size.findtext("height"))

    lines = []

    for obj in root.findall("object"):
        class_name = obj.findtext("name")

        if class_name != "traffic_sign":
            raise ValueError(
                f"Unexpected class '{class_name}' in {xml_path}"
            )

        lines.append(
            convert_bbox(obj, image_width, image_height)
        )

    # Empty annotations are valid negative images.
    # YOLO represents them with an empty label file.
    return lines


def main():
    images = find_images()
    annotations = find_annotations()

    pairs = []

    for filename, xml_path in annotations.items():
        image_path = images.get(filename)

        if image_path is None:
            raise FileNotFoundError(
                f"No image found for annotation: {filename}"
            )

        pairs.append((filename, image_path, xml_path))

    if len(pairs) != 150:
        raise RuntimeError(
            f"Expected 150 image/annotation pairs, found {len(pairs)}"
        )

    random.seed(SEED)
    random.shuffle(pairs)

    train_end = int(len(pairs) * TRAIN_RATIO)
    val_end = train_end + int(len(pairs) * VAL_RATIO)

    splits = {
        "train": pairs[:train_end],
        "val": pairs[train_end:val_end],
        "test": pairs[val_end:],
    }

    if OUTPUT.exists():
        shutil.rmtree(OUTPUT)

    for split in splits:
        (OUTPUT / "images" / split).mkdir(parents=True)
        (OUTPUT / "labels" / split).mkdir(parents=True)

    total_objects = 0

    for split, items in splits.items():
        for filename, image_path, xml_path in items:
            label_lines = process_annotation(xml_path, image_path)

            destination_image = OUTPUT / "images" / split / filename
            destination_label = (
                OUTPUT / "labels" / split / f"{Path(filename).stem}.txt"
            )

            shutil.copy2(image_path, destination_image)

            destination_label.write_text(
                "\n".join(label_lines) + "\n",
                encoding="utf-8",
            )

            total_objects += len(label_lines)

    print("DataCluster → YOLO conversion complete")
    print("======================================")
    print(f"Total images:  {len(pairs)}")
    print(f"Total objects: {total_objects}")
    print(f"Train:         {len(splits['train'])}")
    print(f"Val:           {len(splits['val'])}")
    print(f"Test:          {len(splits['test'])}")
    print(f"Output:        {OUTPUT}")


if __name__ == "__main__":
    main()
