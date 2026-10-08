"""Select Sweetnet and numeric-name melon images, then split by source photo."""

from collections import Counter, defaultdict
import csv
import json
import math
from pathlib import Path
import random
import re
import shutil


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "dataset"
OUTPUT = ROOT / "dataset_sweetnet_numeric"
SEED = 42
SPLITS = ("train", "valid", "test")
RATIOS = (0.7, 0.2, 0.1)
CLASSES = ("Matang", "Mentah", "Setengah Matang")
RF_NAME = re.compile(r"^(?P<stem>.+)_jpg\.rf\.[0-9a-f]{32}\.(?:jpg|jpeg|png)$", re.I)
SWEETNET_NAME = re.compile(r"^Sweetnet_[Dd]ay_\d+_IMG_\d+$", re.I)
NUMERIC_NAME = re.compile(r"^\d+_\d+_[A-Za-z]$")


def classify_name(filename):
    match = RF_NAME.fullmatch(filename)
    if match is None:
        return None
    stem = match.group("stem")
    if SWEETNET_NAME.fullmatch(stem):
        return "Sweetnet", stem
    if NUMERIC_NAME.fullmatch(stem):
        return "Numeric", stem
    return None


def read_records():
    records = []
    families = defaultdict(list)
    output_names = set()
    for original_split in SPLITS:
        images = SOURCE / original_split / "images"
        labels = SOURCE / original_split / "labels"
        for image in sorted(images.iterdir()):
            if not image.is_file():
                continue
            match = classify_name(image.name)
            if match is None:
                continue
            kind, stem = match
            label = labels / (image.stem + ".txt")
            if not label.is_file():
                raise ValueError(f"Missing label: {label}")
            lines = [line.split() for line in label.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
            if not lines or any(len(line) != 5 for line in lines):
                raise ValueError(f"Invalid YOLO detection label: {label}")
            classes = {int(line[0]) for line in lines}
            if len(classes) != 1 or not classes.issubset(range(len(CLASSES))):
                raise ValueError(f"Cannot stratify image with mixed/unknown classes: {label}")
            if image.name.casefold() in output_names:
                raise ValueError(f"Image name repeats across source splits: {image.name}")
            output_names.add(image.name.casefold())
            record = dict(
                kind=kind, stem=stem, class_id=classes.pop(), original_split=original_split,
                image=image, label=label,
            )
            records.append(record)
            families[(kind, stem.casefold())].append(record)

    if len(records) != 2817:
        raise ValueError(f"Expected 2817 selected images, found {len(records)}")
    if not families:
        raise ValueError("No matching images")
    for key, family in families.items():
        if len(family) not in (1, 3):
            raise ValueError(f"Unexpected source-family size {len(family)}: {key}")
        if len({item["class_id"] for item in family}) != 1:
            raise ValueError(f"Conflicting class labels within source family: {key}")
    return records, families


def largest_remainder(total):
    desired = [total * ratio for ratio in RATIOS]
    quotas = [math.floor(value) for value in desired]
    by_remainder = sorted(range(3), key=lambda i: (-(desired[i] - quotas[i]), i))
    for i in by_remainder[: total - sum(quotas)]:
        quotas[i] += 1
    return quotas


def stratum_targets(families):
    strata = defaultdict(list)
    for (kind, _), family in families.items():
        strata[(kind, family[0]["class_id"])].append(family)
    targets = {key: largest_remainder(sum(len(family) for family in grouped))
               for key, grouped in strata.items()}

    overall = largest_remainder(sum(len(family) for grouped in strata.values() for family in grouped))
    current = [sum(row[i] for row in targets.values()) for i in range(3)]
    while current != overall:
        donor = next(i for i in range(3) if current[i] > overall[i])
        receiver = next(i for i in range(3) if current[i] < overall[i])
        candidates = []
        for key, quota in targets.items():
            if quota[donor] == 0:
                continue
            n = sum(quota)
            before = sum((quota[i] - n * RATIOS[i]) ** 2 for i in range(3))
            changed = quota.copy()
            changed[donor] -= 1
            changed[receiver] += 1
            after = sum((changed[i] - n * RATIOS[i]) ** 2 for i in range(3))
            candidates.append((after - before, key))
        if not candidates:
            raise ValueError("Unable to balance split quotas")
        _, chosen = min(candidates)
        targets[chosen][donor] -= 1
        targets[chosen][receiver] += 1
        current[donor] -= 1
        current[receiver] += 1
    return strata, targets, overall


def allocate_families(grouped, target, rng):
    triples = sorted((family for family in grouped if len(family) == 3), key=lambda x: x[0]["stem"])
    singles = sorted((family for family in grouped if len(family) == 1), key=lambda x: x[0]["stem"])
    triple_count, single_count = len(triples), len(singles)
    candidates = []
    for train_triples in range(min(triple_count, target[0] // 3) + 1):
        for valid_triples in range(min(triple_count - train_triples, target[1] // 3) + 1):
            test_triples = triple_count - train_triples - valid_triples
            triple_quota = (train_triples, valid_triples, test_triples)
            if any(3 * triple_quota[i] > target[i] for i in range(3)):
                continue
            single_quota = tuple(target[i] - 3 * triple_quota[i] for i in range(3))
            if sum(single_quota) != single_count:
                continue
            cost = sum((triple_quota[i] - triple_count * RATIOS[i]) ** 2
                       + (single_quota[i] - single_count * RATIOS[i]) ** 2
                       for i in range(3))
            candidates.append((cost, triple_quota, single_quota))
    if not candidates:
        raise ValueError(f"Cannot allocate family sizes to split target {target}")
    _, triple_quota, single_quota = min(candidates)
    rng.shuffle(triples)
    rng.shuffle(singles)
    assigned = {}
    for group, quota in ((triples, triple_quota), (singles, single_quota)):
        offset = 0
        for split, count in zip(SPLITS, quota):
            for family in group[offset: offset + count]:
                assigned[(family[0]["kind"], family[0]["stem"].casefold())] = split
            offset += count
    return assigned


def main():
    if OUTPUT.exists():
        raise FileExistsError(f"Output exists; refusing to overwrite: {OUTPUT}")
    records, families = read_records()
    strata, targets, overall = stratum_targets(families)
    rng = random.Random(SEED)
    assignment = {}
    for key in sorted(strata):
        assignment.update(allocate_families(strata[key], targets[key], rng))
    if len(assignment) != len(families):
        raise AssertionError("Not every source family was assigned")

    image_counts = Counter()
    family_counts = Counter()
    for family_key, split in assignment.items():
        family_counts[split] += 1
        image_counts[split] += len(families[family_key])
    if [image_counts[split] for split in SPLITS] != overall:
        raise AssertionError("Split counts do not match the target")

    for split in SPLITS:
        (OUTPUT / split / "images").mkdir(parents=True)
        (OUTPUT / split / "labels").mkdir(parents=True)

    manifest = []
    class_counts = defaultdict(Counter)
    kind_counts = defaultdict(Counter)
    for record in sorted(records, key=lambda r: (r["kind"], r["stem"], r["image"].name)):
        split = assignment[(record["kind"], record["stem"].casefold())]
        image_output = OUTPUT / split / "images" / record["image"].name
        label_output = OUTPUT / split / "labels" / record["label"].name
        shutil.copy2(record["image"], image_output)
        shutil.copy2(record["label"], label_output)
        class_counts[split][CLASSES[record["class_id"]]] += 1
        kind_counts[split][record["kind"]] += 1
        manifest.append(dict(
            split=split, source_split=record["original_split"], variety_group=record["kind"],
            source_family=record["stem"], class_id=record["class_id"],
            class_name=CLASSES[record["class_id"]],
            source_image=record["image"].relative_to(ROOT).as_posix(),
            output_image=image_output.relative_to(ROOT).as_posix(),
            output_label=label_output.relative_to(ROOT).as_posix(),
        ))

    (OUTPUT / "data.yaml").write_text(
        "train: train/images\nval: valid/images\ntest: test/images\n"
        "nc: 3\nnames: ['Matang', 'Mentah', 'Setengah Matang']\n", encoding="utf-8"
    )
    with (OUTPUT / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest[0]))
        writer.writeheader()
        writer.writerows(manifest)
    summary = dict(
        seed=SEED, selection="Sweetnet filenames and numeric-code filenames from all original splits",
        source="dataset", total_images=len(records), total_source_families=len(families),
        split_ratios=dict(zip(SPLITS, RATIOS)),
        image_counts=dict(image_counts), family_counts=dict(family_counts),
        variety_group_counts={split: dict(kind_counts[split]) for split in SPLITS},
        class_counts={split: dict(class_counts[split]) for split in SPLITS},
        family_overlap_between_splits=False,
    )
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
