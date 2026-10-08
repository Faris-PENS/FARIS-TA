"""Evaluate the frozen YOLO checkpoint on test and export an NMS-free ONNX file."""

import json
import os
from pathlib import Path
import shutil


ROOT = Path(__file__).resolve().parent
CONFIG_DIR = ROOT / ".yolo_config"
CONFIG_DIR.mkdir(exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", str(CONFIG_DIR))

import onnx  # noqa: E402
import onnxruntime as ort  # noqa: E402
from ultralytics import YOLO  # noqa: E402


def main():
    run = ROOT / "runs" / "yolo26n_melon_seed42"
    checkpoint = run / "weights" / "best.pt"
    dataset = ROOT / "dataset_sweetnet_numeric_yolo" / "data.yaml"
    output = ROOT / "yolo_melon.onnx"
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    model = YOLO(str(checkpoint))
    metrics = model.val(
        data=str(dataset), split="test", imgsz=512, batch=16,
        device=0, workers=0, nms=False, plots=True,
        project=str(run), name="test_nms_free", exist_ok=False,
    )
    result = {
        "checkpoint": str(checkpoint),
        "split": "test",
        "inference_head": "one-to-one NMS-free",
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "map50": float(metrics.box.map50),
        "map50_95": float(metrics.box.map),
        "images": 282,
        "note": "Roboflow augmentation is present in test; source-photo families do not cross splits.",
    }
    (run / "test_nms_free_metrics.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    exported = Path(model.export(format="onnx", imgsz=512, batch=1, nms=False,
                                 simplify=False, opset=18, device="cpu"))
    shutil.copy2(exported, output)
    onnx.checker.check_model(str(output))
    session = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"])
    input_shape = session.get_inputs()[0].shape
    output_shape = session.get_outputs()[0].shape
    if input_shape != [1, 3, 512, 512] or output_shape[-1] != 6:
        raise AssertionError(f"Unexpected ONNX shapes: {input_shape} -> {output_shape}")
    print(json.dumps({**result, "onnx": str(output),
                      "input_shape": input_shape, "output_shape": output_shape}, indent=2))


if __name__ == "__main__":
    main()
