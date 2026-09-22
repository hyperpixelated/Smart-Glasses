from pathlib import Path
import zipfile
import shutil
import random

ZIP_PATH = Path.home() / "Downloads" / "archive.zip"
OUTPUT_ROOT = Path("data/processed/sign_classifier")

CLASSES = {
    1: "no_entry",
    15: "no_left_turn",
    16: "no_right_turn",
    18: "speed_90",
    19: "speed_110",
    23: "left_turn",
    24: "right_turn",
    44: "roundabout",
    45: "guarded_railway_crossing",
    46: "unguarded_railway_crossing",
    51: "parking",
    52: "bus_stop",
}

random.seed(42)

with zipfile.ZipFile(ZIP_PATH, "r") as z:
    for split in ["train", "val", "test"]:
        for class_name in CLASSES.values():
            (OUTPUT_ROOT / split / class_name).mkdir(
                parents=True,
                exist_ok=True
            )

    for class_id, class_name in CLASSES.items():
        prefix = f"Indian-Traffic Sign-Dataset/Images/{class_id}/"

        images = [
            name for name in z.namelist()
            if name.startswith(prefix) and name.lower().endswith(".png")
        ]

        if not images:
            print(f"WARNING: no images found for class {class_id}")
            continue

        random.shuffle(images)

        n = len(images)
        train_end = int(0.70 * n)
        val_end = int(0.85 * n)

        splits = {
            "train": images[:train_end],
            "val": images[train_end:val_end],
            "test": images[val_end:],
        }

        for split, split_images in splits.items():
            destination = OUTPUT_ROOT / split / class_name

            for image in split_images:
                filename = Path(image).name
                output_file = destination / filename

                with z.open(image) as source, open(output_file, "wb") as target:
                    shutil.copyfileobj(source, target)

        print(
            f"{class_id:2d} {class_name:30s} "
            f"{len(images):4d} images"
        )

print("\nDataset preparation complete.")
print(f"Output: {OUTPUT_ROOT}")
