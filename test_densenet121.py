"""Predict melon maturity from an image path using the trained DenseNet121.

Examples:
    python test_densenet121.py
    python test_densenet121.py "C:\\path\\to\\melon.jpg"
    python test_densenet121.py "C:\\path\\to\\scene.jpg" --bbox 0.5 0.5 0.4 0.6
    python test_densenet121.py "C:\\path\\to\\melon.jpg" --label "C:\\path\\to\\melon.txt"

For an image in a YOLO images/ folder, the matching labels/ file is detected
automatically. Otherwise the whole image is used unless --bbox or --label is set.
The model stays loaded and accepts more image paths until Ctrl+C is pressed.
"""

import argparse
import math
from pathlib import Path

from PIL import Image, ImageOps
import torch
from torch import nn
from torchvision import models, transforms


ROOT = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = ROOT / "runs" / "densenet121_seed42" / "best_densenet121.pth"


def parse_bbox(values):
    if len(values) != 4:
        raise ValueError("Bounding box must have x_center y_center width height")
    bbox = tuple(float(value) for value in values)
    x, y, width, height = bbox
    if not (all(math.isfinite(value) for value in bbox)
            and 0 <= x <= 1 and 0 <= y <= 1 and 0 < width <= 1 and 0 < height <= 1):
        raise ValueError("YOLO bbox values must be normalized to 0–1 with positive width/height")
    return bbox


def read_label(label_path):
    if not label_path.is_file():
        raise FileNotFoundError(f"YOLO label not found: {label_path}")
    lines = [line.split() for line in label_path.read_text(encoding="utf-8-sig").splitlines()
             if line.strip()]
    if len(lines) != 1 or len(lines[0]) != 5:
        raise ValueError(f"Expected exactly one YOLO detection in {label_path}")
    class_id = int(lines[0][0])
    return parse_bbox(lines[0][1:]), class_id


def find_auto_label(image_path):
    if image_path.parent.name.casefold() != "images":
        return None
    candidate = image_path.parent.parent / "labels" / (image_path.stem + ".txt")
    return candidate if candidate.is_file() else None


def preprocess(image_path, bbox, config):
    image_size = int(config["image_size"])
    padding = float(config["crop_padding_per_side"])
    with Image.open(image_path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
        if bbox is not None:
            x, y, width, height = bbox
            image_width, image_height = image.size
            box = (
                max(0, round((x - (0.5 + padding) * width) * image_width)),
                max(0, round((y - (0.5 + padding) * height) * image_height)),
                min(image_width, round((x + (0.5 + padding) * width) * image_width)),
                min(image_height, round((y + (0.5 + padding) * height) * image_height)),
            )
            if box[2] <= box[0] or box[3] <= box[1]:
                raise ValueError("Bounding box produces an empty crop")
            image = image.crop(box)
        image = image.resize((image_size, image_size), Image.Resampling.LANCZOS)
    normalize = transforms.Normalize(mean=config["mean"], std=config["std"])
    return normalize(transforms.ToTensor()(image)).unsqueeze(0)


def load_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    classes = tuple(checkpoint["classes"])
    model = models.densenet121(weights=None)
    model.classifier = nn.Linear(model.classifier.in_features, len(classes))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    return model, classes, checkpoint["config"]


def predict_image(image_path, args, model, classes, config, device):
    if not image_path.is_file():
        raise FileNotFoundError(f"Gambar tidak ditemukan: {image_path}")

    bbox = None
    true_class_id = None
    if args.bbox is not None:
        bbox = parse_bbox(args.bbox)
    else:
        label_path = args.label.resolve() if args.label else find_auto_label(image_path)
        if label_path is not None:
            bbox, true_class_id = read_label(label_path)
    if true_class_id is not None and not 0 <= true_class_id < len(classes):
        raise ValueError(f"Unknown class ID in label: {true_class_id}")

    tensor = preprocess(image_path, bbox, config).to(device)
    with torch.inference_mode():
        probabilities = model(tensor).softmax(dim=1)[0].cpu().tolist()
    predicted = max(range(len(classes)), key=lambda index: probabilities[index])
    print(f"Prediksi kelas: {classes[predicted]}")
    print(f"Confidence: {probabilities[predicted]:.2%}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", nargs="?", type=Path, help="Path to a melon image")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    location = parser.add_mutually_exclusive_group()
    location.add_argument("--bbox", nargs=4, type=float,
                          metavar=("X_CENTER", "Y_CENTER", "WIDTH", "HEIGHT"),
                          help="Normalized YOLO bounding box for a full scene")
    location.add_argument("--label", type=Path, help="YOLO .txt label containing one box")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    checkpoint_path = args.checkpoint.resolve()
    if not checkpoint_path.is_file():
        parser.error(f"Checkpoint not found: {checkpoint_path}")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is unavailable")
    if args.bbox is not None:
        try:
            parse_bbox(args.bbox)
        except ValueError as error:
            parser.error(str(error))

    device = torch.device("cuda" if args.device == "cuda" or
                          (args.device == "auto" and torch.cuda.is_available()) else "cpu")
    model, classes, config = load_model(checkpoint_path, device)
    next_image = args.image
    try:
        while True:
            if next_image is None:
                entered_path = input("\nMasukkan path gambar (Ctrl+C untuk keluar): ").strip()
                if len(entered_path) >= 2 and entered_path[0] == entered_path[-1] == '"':
                    entered_path = entered_path[1:-1]
                if not entered_path:
                    print("Path gambar harus diisi.")
                    continue
                image_path = Path(entered_path).resolve()
            else:
                image_path = next_image.resolve()
                next_image = None
            try:
                predict_image(image_path, args, model, classes, config, device)
            except (OSError, ValueError) as error:
                print(f"Error: {error}")
    except (KeyboardInterrupt, EOFError):
        print("\nProgram dihentikan.")


if __name__ == "__main__":
    main()
