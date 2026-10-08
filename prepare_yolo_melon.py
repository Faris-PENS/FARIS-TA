"""Build a one-class YOLO detection dataset from the DenseNet split."""

import csv
import json
import math
from pathlib import Path
import shutil


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "dataset_sweetnet_numeric"
OUTPUT = ROOT / "dataset_sweetnet_numeric_yolo"
SPLITS = ("train", "valid", "test")


def main():
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUTPUT}")
    with (SOURCE / "manifest.csv").open(newline="", encoding="utf-8") as handle:
        records = list(csv.DictReader(handle))
    summary = json.loads((SOURCE / "summary.json").read_text(encoding="utf-8"))
    if len(records) != summary["total_images"]:
        raise ValueError("Manifest count differs from source summary")

    seen_names = set()
    families = {}
    checked = []
    counts = {split: 0 for split in SPLITS}
    boxes = {split: 0 for split in SPLITS}
    for row in records:
        split = row["split"]
        if split not in SPLITS:
            raise ValueError(f"Unexpected split: {split}")
        family = (row["variety_group"], row["source_family"].casefold())
        if family in families and families[family] != split:
            raise ValueError(f"Source family crosses splits: {family}")
        families[family] = split
        image = ROOT / row["output_image"]
        label = ROOT / row["output_label"]
        if not image.is_file() or not label.is_file():
            raise FileNotFoundError(f"Missing source pair: {image}, {label}")
        key = (split, image.name.casefold())
        if key in seen_names:
            raise ValueError(f"Duplicate image name: {image}")
        seen_names.add(key)
        lines = [line.split() for line in label.read_text(encoding="utf-8-sig").splitlines()
                 if line.strip()]
        if len(lines) != 1 or len(lines[0]) != 5:
            raise ValueError(f"Expected one YOLO box in {label}")
        class_id = int(lines[0][0])
        if class_id != int(row["class_id"]):
            raise ValueError(f"Class mismatch in {label}")
        x, y, width, height = map(float, lines[0][1:])
        if not (all(map(math.isfinite, (x, y, width, height)))
                and 0 <= x <= 1 and 0 <= y <= 1
                and 0 < width <= 1 and 0 < height <= 1):
            raise ValueError(f"Invalid YOLO box in {label}")
        checked.append((split, image, label, lines[0][1:]))
        counts[split] += 1
        boxes[split] += len(lines)

    if counts != summary["image_counts"]:
        raise ValueError(f"Unexpected split counts: {counts}")
    for split in SPLITS:
        (OUTPUT / split / "images").mkdir(parents=True)
        (OUTPUT / split / "labels").mkdir(parents=True)
    for split, image, label, coordinates in checked:
        shutil.copy2(image, OUTPUT / split / "images" / image.name)
        (OUTPUT / split / "labels" / label.name).write_text(
            "0 " + " ".join(coordinates) + "\n", encoding="utf-8"
        )
    (OUTPUT / "data.yaml").write_text(
        f"path: {OUTPUT.as_posix()}\n"
        "train: train/images\nval: valid/images\ntest: test/images\n"
        "nc: 1\nnames: ['melon']\n", encoding="utf-8"
    )
    (OUTPUT / "summary.json").write_text(json.dumps({
        "source": str(SOURCE), "source_manifest": str(SOURCE / "manifest.csv"),
        "class_mapping": {"Matang": "melon", "Mentah": "melon", "Setengah Matang": "melon"},
        "image_counts": counts, "box_counts": boxes,
        "source_family_overlap_between_splits": False,
        "note": "Original Roboflow augmentation is present in all splits."
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Created {OUTPUT} with {len(checked)} images and one melon box per image")


if __name__ == "__main__":
    main()
