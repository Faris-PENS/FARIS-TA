"""Train a one-class melon detector on the DenseNet Sweetnet + numeric split."""

import os
from pathlib import Path


ROOT = Path(__file__).resolve().parent
CONFIG_DIR = ROOT / ".yolo_config"
CONFIG_DIR.mkdir(exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", str(CONFIG_DIR))

from ultralytics import YOLO  # noqa: E402


def main():
    dataset = ROOT / "dataset_sweetnet_numeric_yolo" / "data.yaml"
    pretrained = ROOT / "yolo26n_pretrained.pt"
    run = ROOT / "runs" / "yolo26n_melon_seed42"
    if not dataset.is_file() or not pretrained.is_file():
        raise FileNotFoundError("Prepare the one-class dataset and pretrained weights first")
    if run.exists():
        raise FileExistsError(f"Refusing to overwrite training run: {run}")
    model = YOLO(str(pretrained))
    model.train(
        data=str(dataset), project=str(ROOT / "runs"), name=run.name,
        epochs=30, patience=8, imgsz=512, batch=16, device=0,
        workers=0, seed=42, deterministic=True, cache=False,
        mosaic=0.0, mixup=0.0, copy_paste=0.0,
        hsv_h=0.0, hsv_s=0.0, hsv_v=0.0,
        degrees=0.0, translate=0.0, scale=0.0, shear=0.0,
        perspective=0.0, fliplr=0.0, flipud=0.0,
        plots=True, verbose=True,
    )
    print(f"Best detector: {run / 'weights' / 'best.pt'}")


if __name__ == "__main__":
    main()
