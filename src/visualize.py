"""Render PaDiM predictions, heatmaps, and predicted defect boundaries.

Panels use the 224x224 evaluation resolution. Labels are normal/anomaly;
PaDiM does not predict named defect types. Scores are raw distances.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import label
import torch

from augmented_dataset import make_dataset
from mvtec_dataset import CATEGORIES, IMAGENET_MEAN, IMAGENET_STD
from padim_features import PaDiMFeatureExtractor, select_device
from padim_inference import PaDiMPredictor


def choose_samples(dataset, predictions, limit):
    """Choose normals, then one example per defect type, then errors/remaining."""
    if limit < 2:
        return list(range(min(limit, len(dataset))))
    indices = []
    normals = [i for i, row in enumerate(dataset.rows) if int(row["label"]) == 0]
    indices.extend(normals[:min(2, limit // 2)])
    defects = sorted({row["defect_type"] for row in dataset.rows if int(row["label"]) == 1})
    for defect in defects:
        index = next(i for i, row in enumerate(dataset.rows) if row["defect_type"] == defect)
        if len(indices) < limit:
            indices.append(index)
    errors = [i for i, row in enumerate(dataset.rows)
              if row["image_path"] in predictions and
              predictions[row["image_path"]]["true_label"] != predictions[row["image_path"]]["predicted_label"]]
    for i in errors + list(range(len(dataset))):
        if len(indices) >= limit:
            break
        if i not in indices:
            indices.append(i)
    return indices


def render(sample, score, anomaly_map, thresholds, accuracy, category, target, min_area):
    mean = np.array(IMAGENET_MEAN)[None, None, :]
    std = np.array(IMAGENET_STD)[None, None, :]
    rgb = np.clip(sample["image"].permute(1, 2, 0).numpy() * std + mean, 0, 1)
    binary = anomaly_map > thresholds["pixel_threshold"]
    components, _ = label(binary, structure=np.ones((3, 3), dtype=np.uint8))
    # Remove tiny components only for display; evaluation metrics stay unchanged.
    sizes = np.bincount(components.ravel())
    keep = sizes >= min_area
    keep[0] = False
    boundary_mask = keep[components]
    region_count = int(keep.sum())
    predicted = "anomaly" if score > thresholds["image_threshold"] else "normal"
    actual = "anomaly" if sample["label"] else "normal"
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.1))
    axes[0].imshow(rgb)
    axes[0].set_title("Input (evaluation resolution)")
    axes[1].imshow(rgb)
    heat = axes[1].imshow(anomaly_map, cmap="jet", alpha=0.55, vmin=0,
                          vmax=max(thresholds["pixel_threshold"] * 2, 1e-6))
    axes[1].set_title("Anomaly heatmap (raw distance)")
    fig.colorbar(heat, ax=axes[1], fraction=0.046, pad=0.04)
    axes[2].imshow(rgb)
    if boundary_mask.any():
        # Padding closes contours for defects that touch the image edge.
        padded = np.pad(boundary_mask.astype(float), 1)
        ys = np.arange(-1, boundary_mask.shape[0] + 1)
        xs = np.arange(-1, boundary_mask.shape[1] + 1)
        axes[2].contour(xs, ys, padded, levels=[0.5], colors="red", linewidths=1.4)
        axes[2].set_xlim(-0.5, boundary_mask.shape[1] - 0.5)
        axes[2].set_ylim(boundary_mask.shape[0] - 0.5, -0.5)
    axes[2].set_title("Predicted defect boundaries (red)")
    accuracy_text = f"Category test accuracy: {accuracy:.1%}" if accuracy is not None else ""
    axes[2].text(0.98, 0.98, f"{category} / {predicted}\nScore: {score:.2f}\n{accuracy_text}",
                 transform=axes[2].transAxes, ha="right", va="top", fontsize=7,
                 color="white", bbox=dict(facecolor="black", alpha=0.7, pad=3))
    axes[3].imshow(sample["mask"][0].numpy(), cmap="gray", vmin=0, vmax=1)
    axes[3].set_title(f"Ground truth: {sample['defect_type']}")
    for ax in axes:
        ax.set_axis_off()
    fig.suptitle(f"{sample['image_path']} | Actual: {actual} | Predicted: {predicted}", fontsize=10)
    fig.text(0.5, 0.01,
             f"Image threshold: {thresholds['image_threshold']:.2f} | Pixel threshold: {thresholds['pixel_threshold']:.2f} | "
             f"Contours: components >= {min_area} pixels. Image decisions and contours use separate thresholds.",
             ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0.05, 1, 0.93))
    fig.savefig(target, dpi=160)
    plt.close(fig)
    return predicted, region_count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--splits-dir", type=Path, default=Path("splits"))
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    parser.add_argument("--calibration-dir", type=Path, default=Path("results/calibration"))
    parser.add_argument("--evaluation-dir", type=Path, default=Path("results/evaluation"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/visualizations_boundaries"))
    parser.add_argument("--category", choices=["all"] + CATEGORIES, default="all")
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--samples-per-category", type=int, default=6)
    parser.add_argument("--min-area", type=int, default=16)
    args = parser.parse_args()
    if min(args.samples_per_category, args.min_area) < 1:
        parser.error("Sample count and minimum area must be positive")
    torch.set_num_threads(4)
    device = select_device(args.device)
    categories = CATEGORIES if args.category == "all" else [args.category]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    extractor = None
    rows = []
    for category in categories:
        model_path = args.models_dir / f"{category}_padim.pt"
        thresholds = json.loads((args.calibration_dir / f"{category}_thresholds.json").read_text())
        digest = hashlib.sha256()
        with model_path.open("rb") as file:
            for block in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(block)
        if thresholds["category"] != category or digest.hexdigest() != thresholds["model_sha256"]:
            raise ValueError(f"Calibration/model mismatch: {category}")
        if extractor is None:
            config = torch.load(model_path, map_location="cpu", weights_only=True)
            extractor = PaDiMFeatureExtractor(config["n_features"], config["seed"]).to(device).eval()
            del config
        predictor = PaDiMPredictor(model_path, device=device, extractor=extractor,
                                    sigma=thresholds["gaussian_sigma"])
        dataset = make_dataset(args.data_root, args.splits_dir, category, "test",
                               image_size=predictor.image_size, augment=False)
        metrics = json.loads((args.evaluation_dir / f"{category}_metrics.json").read_text())
        if metrics["model_sha256"] != thresholds["model_sha256"] or metrics["image_threshold"] != thresholds["image_threshold"]:
            raise ValueError(f"Evaluation/model mismatch: {category}")
        with (args.evaluation_dir / f"{category}_predictions.csv").open(newline="") as file:
            predictions = {row["image_path"]: row for row in csv.DictReader(file)}
        chosen = choose_samples(dataset, predictions, args.samples_per_category)
        directory = args.output_dir / category
        directory.mkdir(exist_ok=True)
        for index in chosen:
            sample = dataset[index]
            scores, maps = predictor.predict(sample["image"].unsqueeze(0))
            target = directory / f"{sample['defect_type']}_{Path(sample['image_path']).stem}.png"
            predicted, regions = render(sample, float(scores[0]), maps[0], thresholds,
                                      metrics["image"]["accuracy"], category, target, args.min_area)
            rows.append(dict(category=category, image_path=sample["image_path"],
                             true_label=sample["label"], predicted_label=predicted,
                             anomaly_score=float(scores[0]), regions=regions,
                             visualization_path=target.relative_to(args.output_dir).as_posix()))
        print(f"{category:12} saved {len(chosen)} boundary prediction panels", flush=True)
        del predictor
    with (args.output_dir / "boundary_visualization_index.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved {len(rows)} boundary prediction panels to {args.output_dir}", flush=True)
    print("These are selected examples; full-test metrics are in results/evaluation.", flush=True)


if __name__ == "__main__":
    main()
