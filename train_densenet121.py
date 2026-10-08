"""Train DenseNet121 on the Sweetnet + numeric-code melon split.

Run from any directory:
    python train_densenet121.py
    python train_densenet121.py --epochs 30 --batch-size 32

The Roboflow archive already contains augmented images, so this script does not
add online augmentation. Each YOLO box is cropped with 5% padding per side.
"""

import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import random
import time

import numpy as np
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET = ROOT / "dataset_sweetnet_numeric"
CLASSES = ("Matang", "Mentah", "Setengah Matang")
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
IMAGE_SIZE = 224
SEED = 42


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def seed_everything():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def read_manifest(dataset_dir):
    with (dataset_dir / "manifest.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("Empty dataset manifest")
    summary = json.loads((dataset_dir / "summary.json").read_text(encoding="utf-8"))
    if len(rows) != summary["total_images"]:
        raise ValueError("Manifest size differs from dataset summary")
    source_splits = defaultdict(set)
    for row in rows:
        if row["split"] not in ("train", "valid", "test"):
            raise ValueError(f"Unknown split: {row['split']}")
        if row["variety_group"] not in ("Sweetnet", "Numeric"):
            raise ValueError(f"Unexpected variety group: {row['variety_group']}")
        source_splits[(row["variety_group"], row["source_family"].casefold())].add(row["split"])
    if any(len(splits) != 1 for splits in source_splits.values()):
        raise ValueError("A source-photo family occurs in multiple splits")
    return rows, summary


class MelonCrops(Dataset):
    def __init__(self, dataset_dir, rows, split):
        self.samples = []
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD),
        ])
        for row in rows:
            if row["split"] != split:
                continue
            image = dataset_dir / split / "images" / Path(row["output_image"]).name
            label_file = dataset_dir / split / "labels" / Path(row["output_label"]).name
            if not image.is_file() or not label_file.is_file():
                raise FileNotFoundError(f"Missing image or label: {image}, {label_file}")
            lines = [line.split() for line in label_file.read_text(encoding="utf-8-sig").splitlines()
                     if line.strip()]
            if len(lines) != 1 or len(lines[0]) != 5:
                raise ValueError(f"Expected one YOLO box: {label_file}")
            class_id = int(lines[0][0])
            if class_id != int(row["class_id"]) or not 0 <= class_id < len(CLASSES):
                raise ValueError(f"Class mismatch: {label_file}")
            x, y, w, h = (float(value) for value in lines[0][1:])
            if not all(np.isfinite((x, y, w, h))) or not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
                raise ValueError(f"Invalid YOLO coordinates: {label_file}")
            self.samples.append((image, class_id, (x, y, w, h), row["variety_group"]))
        if not self.samples:
            raise ValueError(f"No samples in split {split}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, class_id, (x, y, w, h), group = self.samples[index]
        with Image.open(image_path) as source:
            image = source.convert("RGB")
            width, height = image.size
            box = (
                max(0, round((x - 0.55 * w) * width)),
                max(0, round((y - 0.55 * h) * height)),
                min(width, round((x + 0.55 * w) * width)),
                min(height, round((y + 0.55 * h) * height)),
            )
            if box[2] <= box[0] or box[3] <= box[1]:
                raise ValueError(f"Empty crop for {image_path}")
            crop = image.crop(box).resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.LANCZOS)
        return self.transform(crop), class_id, image_path.name, group


def make_model():
    model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
    model.classifier = nn.Linear(model.classifier.in_features, len(CLASSES))
    return model


def scores(actual, predicted):
    actual = np.asarray(actual)
    predicted = np.asarray(predicted)
    labels = list(range(len(CLASSES)))
    precision, recall, f1, support = precision_recall_fscore_support(
        actual, predicted, labels=labels, zero_division=0
    )
    return dict(
        accuracy=float(accuracy_score(actual, predicted)),
        macro_f1=float(np.mean(f1)),
        confusion_matrix=confusion_matrix(actual, predicted, labels=labels).tolist(),
        per_class={name: dict(precision=float(precision[i]), recall=float(recall[i]),
                              f1=float(f1[i]), support=int(support[i]))
                   for i, name in enumerate(CLASSES)},
    )


def evaluate(model, loader, class_weights, device):
    model.eval()
    loss_numerator = loss_denominator = 0.0
    actual, predicted, probability_rows, filenames, groups = [], [], [], [], []
    with torch.inference_mode():
        for images, labels, batch_names, batch_groups in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(images)
            loss_numerator += nn.functional.cross_entropy(
                logits, labels, weight=class_weights, reduction="sum"
            ).item()
            loss_denominator += class_weights[labels].sum().item()
            probabilities = logits.softmax(dim=1).cpu().numpy()
            actual.extend(labels.cpu().tolist())
            predicted.extend(probabilities.argmax(axis=1).tolist())
            probability_rows.extend(probabilities.tolist())
            filenames.extend(batch_names)
            groups.extend(batch_groups)
    result = scores(actual, predicted)
    result["loss"] = loss_numerator / loss_denominator
    result["by_variety_group"] = {
        group: scores([actual[i] for i in indices], [predicted[i] for i in indices])
        for group in sorted(set(groups))
        if (indices := [i for i, value in enumerate(groups) if value == group])
    }
    return result, list(zip(filenames, groups, actual, predicted, probability_rows))


def train_epoch(model, loader, class_weights, optimizer, scaler, device):
    model.train()
    loss_numerator = loss_denominator = correct = total = 0.0
    for images, labels, _, _ in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            logits = model(images)
            loss_sum = nn.functional.cross_entropy(logits, labels, weight=class_weights, reduction="sum")
            weight_sum = class_weights[labels].sum()
            loss = loss_sum / weight_sum
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite training loss")
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        loss_numerator += loss_sum.item()
        loss_denominator += weight_sum.item()
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += len(labels)
    return dict(loss=loss_numerator / loss_denominator, accuracy=correct / total)


def save_predictions(path, records):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["image", "variety_group", "actual", "predicted"]
                        + [f"prob_{name.replace(' ', '_')}" for name in CLASSES])
        for filename, group, actual, predicted, probabilities in records:
            writer.writerow([filename, group, CLASSES[actual], CLASSES[predicted]] + probabilities)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--run", type=Path, default=ROOT / "runs" / "densenet121_seed42")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--check-only", action="store_true", help="Validate data and one forward pass; do not train")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.learning_rate <= 0 or args.patience < 1:
        parser.error("epochs, batch-size, learning-rate and patience must be positive")
    dataset_dir = args.dataset.resolve()
    run_dir = args.run.resolve()
    rows, summary = read_manifest(dataset_dir)
    seed_everything()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    datasets = {split: MelonCrops(dataset_dir, rows, split) for split in ("train", "valid", "test")}
    for split, dataset in datasets.items():
        if len(dataset) != summary["image_counts"][split]:
            raise ValueError(f"Unexpected {split} sample count")
    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        datasets["train"], batch_size=args.batch_size, shuffle=True, num_workers=0,
        pin_memory=pin_memory, generator=torch.Generator().manual_seed(SEED),
    )
    valid_loader = DataLoader(datasets["valid"], batch_size=args.batch_size,
                              num_workers=0, pin_memory=pin_memory)
    test_loader = DataLoader(datasets["test"], batch_size=args.batch_size,
                             num_workers=0, pin_memory=pin_memory)
    counts = Counter(sample[1] for sample in datasets["train"].samples)
    weights = torch.tensor([len(datasets["train"]) / (len(CLASSES) * counts[i])
                            for i in range(len(CLASSES))], dtype=torch.float32, device=device)
    model = make_model().to(device)
    print(f"Device: {device}; samples: {[len(datasets[s]) for s in ('train', 'valid', 'test')]}", flush=True)
    if args.check_only:
        images, labels, _, _ = next(iter(train_loader))
        with torch.inference_mode():
            logits = model(images.to(device))
        print(f"CHECK OK: batch={len(labels)}, logits={tuple(logits.shape)}", flush=True)
        return
    if run_dir.exists():
        raise FileExistsError(f"Run directory already exists: {run_dir}")
    run_dir.mkdir(parents=True)
    config = dict(model="densenet121", pretrained="IMAGENET1K_V1", seed=SEED,
                  image_size=IMAGE_SIZE, crop_padding_per_side=0.05,
                  crop_resize="PIL LANCZOS", input="RGB YOLO bbox crop", mean=MEAN, std=STD,
                  online_augmentation=False, epochs=args.epochs, batch_size=args.batch_size,
                  learning_rate=args.learning_rate, weight_decay=1e-4,
                  patience=args.patience, checkpoint_selection="highest validation macro F1, then lowest validation loss",
                  device=str(device), dataset=str(dataset_dir),
                  evaluation_note="Roboflow augmentation was already applied; some validation/test images are augmented")
    save_json(run_dir / "config.json", config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )
    scaler = torch.amp.GradScaler("cuda", enabled=pin_memory)
    checkpoint = run_dir / "best_densenet121.pth"
    best_f1, best_loss, wait = -1.0, float("inf"), 0
    history = []
    start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()
        learning_rate = optimizer.param_groups[0]["lr"]
        training = train_epoch(model, train_loader, weights, optimizer, scaler, device)
        validation, _ = evaluate(model, valid_loader, weights, device)
        scheduler.step(validation["loss"])
        better = (validation["macro_f1"] > best_f1 + 1e-12 or
                  (abs(validation["macro_f1"] - best_f1) <= 1e-12 and validation["loss"] < best_loss))
        if better:
            best_f1, best_loss, wait = validation["macro_f1"], validation["loss"], 0
            state = {key: tensor.detach().cpu().clone() for key, tensor in model.state_dict().items()}
            torch.save(dict(model_state_dict=state, classes=CLASSES, epoch=epoch,
                            validation=validation, config=config), checkpoint)
        else:
            wait += 1
        entry = dict(epoch=epoch, train_loss=training["loss"], train_accuracy=training["accuracy"],
                     validation_loss=validation["loss"], validation_accuracy=validation["accuracy"],
                     validation_macro_f1=validation["macro_f1"], learning_rate=learning_rate,
                     seconds=time.perf_counter() - epoch_start, best=better)
        history.append(entry)
        save_json(run_dir / "history.json", history)
        print(f"Epoch {epoch:02d}/{args.epochs}: train acc={entry['train_accuracy']:.3f}, "
              f"valid acc={entry['validation_accuracy']:.3f}, "
              f"valid macro F1={entry['validation_macro_f1']:.3f}, "
              f"valid loss={entry['validation_loss']:.4f}, "
              f"{entry['seconds']:.1f}s {'BEST' if better else f'patience {wait}/{args.patience}'}", flush=True)
        if wait >= args.patience:
            break
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(saved["model_state_dict"])
    validation, _ = evaluate(model, valid_loader, weights, device)
    test, predictions = evaluate(model, test_loader, weights, device)
    save_json(run_dir / "validation_metrics.json", validation)
    save_json(run_dir / "test_metrics.json", test)
    save_predictions(run_dir / "test_predictions.csv", predictions)
    save_json(run_dir / "run_summary.json", dict(best_epoch=saved["epoch"],
              epochs_completed=len(history), training_seconds=time.perf_counter()-start,
              validation_macro_f1=validation["macro_f1"], test_accuracy=test["accuracy"],
              test_macro_f1=test["macro_f1"], checkpoint=str(checkpoint)))
    print(f"Done. Best epoch={saved['epoch']}; test accuracy={test['accuracy']:.3f}; "
          f"test macro F1={test['macro_f1']:.3f}; output={run_dir}", flush=True)


if __name__ == "__main__":
    main()
