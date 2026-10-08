"""Find melon boxes with YOLO ONNX, then classify each crop with DenseNet ONNX."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image, ImageOps

from predict_onnx import CLASSES, preprocess as preprocess_classifier


ROOT = Path(__file__).resolve().parent


def session(path):
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), sess_options=options,
                                providers=["CPUExecutionProvider"])


def detector_input(image, size=512):
    rgb = np.asarray(image, dtype=np.uint8)
    height, width = rgb.shape[:2]
    gain = min(size / height, size / width)
    resized_width, resized_height = round(width * gain), round(height * gain)
    if (resized_width, resized_height) != (width, height):
        rgb = cv2.resize(rgb, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
    half_width = (size - resized_width) / 2
    half_height = (size - resized_height) / 2
    rgb = cv2.copyMakeBorder(
        rgb, round(half_height - 0.1), round(half_height + 0.1),
        round(half_width - 0.1), round(half_width + 0.1),
        cv2.BORDER_CONSTANT, value=(114, 114, 114),
    )
    tensor = np.ascontiguousarray(rgb.transpose(2, 0, 1)[None], dtype=np.float32) / 255.0
    return tensor, gain, half_width, half_height


def predict(image_path, yolo, classifier, conf=0.25):
    with Image.open(image_path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
    width, height = image.size
    tensor, gain, pad_x, pad_y = detector_input(image)
    rows = yolo.run(None, {yolo.get_inputs()[0].name: tensor})[0]
    if rows.ndim != 3 or rows.shape[0] != 1 or rows.shape[2] != 6:
        raise ValueError(f"Expected YOLO26 NMS-free output [1, N, 6], got {rows.shape}")

    boxes = []
    for x1, y1, x2, y2, score, class_id in sorted(rows[0], key=lambda row: -row[4]):
        if score < conf:
            break
        if int(class_id) != 0:
            continue
        left = max(0.0, min(float((x1 - pad_x) / gain), width))
        top = max(0.0, min(float((y1 - pad_y) / gain), height))
        right = max(0.0, min(float((x2 - pad_x) / gain), width))
        bottom = max(0.0, min(float((y2 - pad_y) / gain), height))
        if right <= left or bottom <= top:
            continue
        boxes.append((left, top, right, bottom, float(score)))

    detections = []
    for left, top, right, bottom, score in boxes:
        bbox = ((left + right) / (2 * width), (top + bottom) / (2 * height),
                (right - left) / width, (bottom - top) / height)
        crop = preprocess_classifier(image_path, bbox)
        logits = classifier.run(["logits"], {"input": crop})[0][0]
        probabilities = np.exp(logits - logits.max())
        probabilities /= probabilities.sum()
        predicted = int(probabilities.argmax())
        detections.append({
            "box_xyxy": [round(v) for v in (left, top, right, bottom)],
            "detection_confidence": score,
            "predicted_class": CLASSES[predicted],
            "classification_confidence": float(probabilities[predicted]),
            "probabilities": {name: float(probabilities[index])
                              for index, name in enumerate(CLASSES)},
        })
    return {"image": str(image_path), "detections": detections}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--detector", type=Path, default=ROOT / "yolo_melon.onnx")
    parser.add_argument("--classifier", type=Path, default=ROOT / "densenet121.onnx")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="Minimum YOLO detection confidence (default: 0.25)")
    args = parser.parse_args()
    if not 0 <= args.conf <= 1:
        parser.error("--conf must be between 0 and 1")
    for path in (args.image, args.detector, args.classifier):
        if not path.is_file():
            parser.error(f"File not found: {path}")
    result = predict(args.image, session(args.detector), session(args.classifier), args.conf)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
