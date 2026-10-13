"""Create reproducible PaDiM split manifests without moving dataset files."""
import argparse
import csv
import random
from pathlib import Path

CATEGORIES = ["bottle", "cable", "capsule", "carpet", "grid", "hazelnut",
              "leather", "metal_nut", "pill", "screw", "tile", "toothbrush",
              "transistor", "wood", "zipper"]
FIELDS = ["category", "split", "image_path", "label", "defect_type", "mask_path"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("splits"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    root = args.data_root.expanduser().resolve()
    manifests = {}
    summary = []

    # Validate all categories before writing any manifests.
    for category in CATEGORIES:
        base = root / category
        for folder in ["train/good", "test/good", "ground_truth"]:
            if not (base / folder).is_dir():
                raise SystemExit(f"Missing folder: {base / folder}")
        normal = sorted((base / "train/good").glob("*.png"))
        test = sorted((base / "test").glob("*/*.png"))
        if len(normal) < 2 or not test:
            raise SystemExit(f"Insufficient images: {category}")
        random.Random(args.seed).shuffle(normal)
        n_train = int(0.8 * len(normal))
        groups = {"train": normal[:n_train], "validation": normal[n_train:], "test": test}
        sets = [set(paths) for paths in groups.values()]
        if any(sets[i] & sets[j] for i in range(3) for j in range(i + 1, 3)):
            raise SystemExit(f"Overlapping splits: {category}")

        for split, paths in groups.items():
            rows = []
            for image in paths:
                defect = image.parent.name
                label = int(defect != "good")
                mask_path = ""
                if label:
                    mask = base / "ground_truth" / defect / f"{image.stem}_mask.png"
                    if not mask.is_file():
                        raise SystemExit(f"Missing mask: {mask}")
                    mask_path = mask.relative_to(root).as_posix()
                rows.append(dict(category=category, split=split,
                                 image_path=image.relative_to(root).as_posix(),
                                 label=label, defect_type=defect, mask_path=mask_path))
            manifests[f"{category}_{split}.csv"] = rows
        summary.append(dict(category=category, train=n_train,
                            validation=len(normal) - n_train, test=len(test), seed=args.seed))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for filename, rows in manifests.items():
        with (args.output_dir / filename).open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)
    with (args.output_dir / "split_summary.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["category", "train", "validation", "test", "seed"])
        writer.writeheader()
        writer.writerows(summary)
    for row in summary:
        print(f"{row['category']:12} train={row['train']:3} validation={row['validation']:3} test={row['test']:3}")
    print(f"\nSaved 45 split manifests and split_summary.csv to {args.output_dir.resolve()}")
    print("Labels: 0=normal, 1=anomaly. Paths are relative to --data-root.")
    print("Dataset files unchanged. Rerunning overwrites manifests with the same deterministic split.")


if __name__ == "__main__":
    main()
