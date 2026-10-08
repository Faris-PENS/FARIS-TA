"""Evaluate ONNX YOLO -> DenseNet predictions on the held-out test images."""

import csv
import json
from pathlib import Path

from PIL import Image

from predict_full_image import ROOT, predict, session
from predict_onnx import CLASSES, read_label


def overlap(box_a, box_b):
    left, top = max(box_a[0], box_b[0]), max(box_a[1], box_b[1])
    right, bottom = min(box_a[2], box_b[2]), min(box_a[3], box_b[3])
    intersection = max(0, right - left) * max(0, bottom - top)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    return intersection / (area_a + area_b - intersection)


def main():
    dataset = ROOT / "dataset_sweetnet_numeric"
    run = ROOT / "runs" / "yolo26n_melon_seed42"
    with (dataset / "manifest.csv").open(newline="", encoding="utf-8") as handle:
        records = [row for row in csv.DictReader(handle) if row["split"] == "test"]
    detector = session(ROOT / "yolo_melon.onnx")
    classifier = session(ROOT / "densenet121.onnx")
    predictions = []
    for row in records:
        image = ROOT / row["output_image"]
        label = ROOT / row["output_label"]
        expected = CLASSES[int(row["class_id"])]
        detection = predict(image, detector, classifier)["detections"]
        best = detection[0] if detection else None
        with Image.open(image) as source:
            width, height = source.size
        x, y, box_width, box_height = read_label(label)
        truth = ((x - box_width / 2) * width, (y - box_height / 2) * height,
                 (x + box_width / 2) * width, (y + box_height / 2) * height)
        iou = overlap(best["box_xyxy"], truth) if best else 0.0
        predictions.append({
            "image": row["output_image"], "actual_class": expected,
            "predicted_class": best["predicted_class"] if best else "NO_DETECTION",
            "detection_count": len(detection), "detection_confidence":
                best["detection_confidence"] if best else None,
            "iou_top_box": iou,
            "classification_confidence": best["classification_confidence"] if best else None,
        })
    total = len(predictions)
    detected = [row for row in predictions if row["detection_count"]]
    correct = [row for row in detected if row["predicted_class"] == row["actual_class"]]
    summary = {
        "images": total, "detected_images": len(detected),
        "detection_coverage": len(detected) / total,
        "top_box_iou50_images": sum(row["iou_top_box"] >= 0.5 for row in predictions),
        "classification_accuracy_on_detected": len(correct) / len(detected) if detected else None,
        "end_to_end_accuracy": len(correct) / total,
        "correct_class_and_iou50": sum(row["predicted_class"] == row["actual_class"]
                                       and row["iou_top_box"] >= 0.5 for row in predictions) / total,
        "extra_detections": sum(max(0, row["detection_count"] - 1) for row in predictions),
        "confidence_threshold": 0.25,
        "note": "Test images include existing Roboflow augmentation; each image has one labeled melon.",
    }
    (run / "end_to_end_test.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    with (run / "end_to_end_test_predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(predictions[0]))
        writer.writeheader()
        writer.writerows(predictions)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
