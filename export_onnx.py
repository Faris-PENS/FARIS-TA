"""Export the trained TA-FARIS-2 DenseNet121 checkpoint as a single ONNX file."""

import argparse
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch

from test_densenet121 import DEFAULT_CHECKPOINT, find_auto_label, load_model, preprocess, read_label


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / "densenet121.onnx"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    checkpoint = args.checkpoint.resolve()
    output = args.output.resolve()
    if not checkpoint.is_file():
        parser.error(f"Checkpoint not found: {checkpoint}")

    model, classes, config = load_model(checkpoint, torch.device("cpu"))
    image_size = int(config["image_size"])
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        torch.zeros(1, 3, image_size, image_size),
        str(output),
        export_params=True,
        opset_version=18,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        dynamo=False,
    )

    graph = onnx.load(str(output), load_external_data=False)
    external_files = [entry.value for tensor in graph.graph.initializer
                      for entry in tensor.external_data if entry.key == "location"]
    if external_files:
        raise RuntimeError(f"Export depends on external data files: {external_files}")
    onnx.helper.set_model_props(graph, {
        "classes": json.dumps(classes, ensure_ascii=False),
        "input_preprocessing": "RGB YOLO bbox crop with 5% padding, LANCZOS resize, ImageNet normalization",
        "mean": json.dumps(config["mean"]),
        "std": json.dumps(config["std"]),
    })
    onnx.save(graph, str(output))
    onnx.checker.check_model(str(output))

    image_dir = ROOT / "dataset_sweetnet_numeric" / "test" / "images"
    sample_image = next((path for path in image_dir.iterdir()
                         if path.suffix.lower() in {".jpg", ".jpeg", ".png"}), None)
    if sample_image is None:
        raise FileNotFoundError(f"No test image in {image_dir}")
    label_path = find_auto_label(sample_image)
    bbox, _ = read_label(label_path)
    sample = preprocess(sample_image, bbox, config)

    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    session = ort.InferenceSession(str(output), sess_options=options,
                                   providers=["CPUExecutionProvider"])
    if session.get_inputs()[0].shape[0] != "batch":
        raise AssertionError("ONNX export has no dynamic batch dimension")
    for batch in (sample, torch.cat((sample, sample.flip(-1)), dim=0)):
        with torch.inference_mode():
            expected = model(batch).softmax(dim=1).numpy()
        actual_logits = session.run(["logits"], {"input": batch.numpy()})[0]
        actual = torch.from_numpy(actual_logits).softmax(dim=1).numpy()
        if not np.allclose(expected, actual, rtol=1e-4, atol=1e-5):
            raise AssertionError(f"ONNX differs from PyTorch at batch size {len(batch)}")
        if not np.array_equal(expected.argmax(axis=1), actual.argmax(axis=1)):
            raise AssertionError(f"ONNX class differs from PyTorch at batch size {len(batch)}")

    print(f"ONNX verified: {output} ({output.stat().st_size:,} bytes)")
    print(f"Classes: {', '.join(classes)}")


if __name__ == "__main__":
    main()
