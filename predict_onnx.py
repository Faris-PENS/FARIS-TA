"""Predict melon maturity with the exported ONNX model (CPU only).

An image in a YOLO images/ directory uses its matching labels/ file automatically.
For other images, provide --bbox or --label, or pass an already cropped melon image.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageOps


ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = ROOT / "densenet121.onnx"
CLASSES = ("Matang", "Mentah", "Setengah Matang")
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def parse_bbox(values):
    if len(values) != 4:
        raise ValueError("Bounding box needs x_center y_center width height")
    bbox = tuple(float(value) for value in values)
    x, y, width, height = bbox
    if not (all(math.isfinite(value) for value in bbox)
            and 0 <= x <= 1 and 0 <= y <= 1
            and 0 < width <= 1 and 0 < height <= 1):
        raise ValueError("YOLO bounding box values must be normalized to 0–1")
    return bbox


def read_label(path):
    lines = [line.split() for line in path.read_text(encoding="utf-8-sig").splitlines()
             if line.strip()]
    if len(lines) != 1 or len(lines[0]) != 5:
        raise ValueError(f"Expected one YOLO box in {path}")
    return parse_bbox(lines[0][1:])


def preprocess(path, bbox):
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
        if bbox is not None:
            x, y, width, height = bbox
            image_width, image_height = image.size
            box = (
                max(0, round((x - 0.55 * width) * image_width)),
                max(0, round((y - 0.55 * height) * image_height)),
                min(image_width, round((x + 0.55 * width) * image_width)),
                min(image_height, round((y + 0.55 * height) * image_height)),
            )
            if box[2] <= box[0] or box[3] <= box[1]:
                raise ValueError("Bounding box produces an empty crop")
            image = image.crop(box)
        image = image.resize((224, 224), Image.Resampling.LANCZOS)
        array = np.asarray(image, dtype=np.float32) / 255.0
    normalized = (array - MEAN) / STD
    return np.ascontiguousarray(normalized.transpose(2, 0, 1)[None])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    location = parser.add_mutually_exclusive_group()
    location.add_argument("--bbox", nargs=4, type=float,
                          metavar=("X_CENTER", "Y_CENTER", "WIDTH", "HEIGHT"))
    location.add_argument("--label", type=Path, help="YOLO label file")
    args = parser.parse_args()

    image = args.image.resolve()
    model = args.model.resolve()
    if not image.is_file() or not model.is_file():
        parser.error(f"Image or ONNX model not found: {image}, {model}")
    try:
        if args.bbox is not None:
            bbox = parse_bbox(args.bbox)
        else:
            label = args.label.resolve() if args.label else None
            if label is None and image.parent.name.casefold() == "images":
                candidate = image.parent.parent / "labels" / f"{image.stem}.txt"
                label = candidate if candidate.is_file() else None
            bbox = read_label(label) if label is not None else None
        input_array = preprocess(image, bbox)
    except (OSError, ValueError) as error:
        parser.error(str(error))

    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(model), sess_options=options,
                                   providers=["CPUExecutionProvider"])
    logits = session.run(["logits"], {"input": input_array})[0][0]
    probabilities = np.exp(logits - logits.max())
    probabilities /= probabilities.sum()
    predicted = int(probabilities.argmax())
    print(json.dumps({
        "predicted_class": CLASSES[predicted],
        "confidence": float(probabilities[predicted]),
        "probabilities": {name: float(probabilities[index])
                          for index, name in enumerate(CLASSES)},
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
