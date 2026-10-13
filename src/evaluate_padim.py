"""Evaluate fitted PaDiM models on untouched official test sets.

Uses fixed validation-calibrated thresholds; never tunes on test data.
Place this file in src beside padim_inference.py and the dataset modules.
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
import torch
from sklearn.metrics import (accuracy_score, average_precision_score,
                             confusion_matrix, precision_recall_curve,
                             precision_recall_fscore_support, roc_auc_score, roc_curve)
from torch.utils.data import DataLoader

from augmented_dataset import make_dataset
from mvtec_dataset import CATEGORIES
from padim_features import PaDiMFeatureExtractor, select_device
from padim_inference import PaDiMPredictor


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def binary_metrics(labels, scores, threshold):
    prediction = scores > threshold
    p, r, f1, _ = precision_recall_fscore_support(labels, prediction, average="binary", zero_division=0)
    tn, fp, fn, tp = confusion_matrix(labels, prediction, labels=[0, 1]).ravel()
    return dict(auroc=float(roc_auc_score(labels, scores)),
                average_precision=float(average_precision_score(labels, scores)),
                accuracy=float(accuracy_score(labels, prediction)),
                precision=float(p), recall=float(r), f1=float(f1),
                tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp))


def plot_curves(category, labels, scores, metrics, directory):
    fpr, tpr, _ = roc_curve(labels, scores)
    precision, recall, _ = precision_recall_curve(labels, scores)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    axes[0].plot(fpr, tpr, label=f"AUROC = {metrics['auroc']:.3f}")
    axes[0].plot([0, 1], [0, 1], "--", color="gray")
    axes[0].set(xlabel="False positive rate", ylabel="True positive rate", title="Image ROC")
    axes[0].legend()
    axes[1].plot(recall, precision, label=f"AP = {metrics['average_precision']:.3f}")
    axes[1].axhline(float(np.mean(labels)), linestyle="--", color="gray", label="Anomaly prevalence")
    axes[1].set(xlabel="Recall", ylabel="Precision", title="Image precision–recall")
    axes[1].legend()
    matrix = np.array([[metrics['tn'], metrics['fp']], [metrics['fn'], metrics['tp']]])
    axes[2].imshow(matrix, cmap="Blues")
    for (i, j), value in np.ndenumerate(matrix):
        axes[2].text(j, i, str(value), ha="center", va="center",
                     color="white" if value > matrix.max() / 2 else "black")
    axes[2].set(xticks=[0, 1], yticks=[0, 1], xticklabels=["Normal", "Anomaly"],
                yticklabels=["Normal", "Anomaly"], xlabel="Predicted", ylabel="Actual",
                title="Fixed-threshold confusion matrix")
    fig.suptitle(category)
    fig.tight_layout()
    fig.savefig(directory / f"{category}_image_metrics.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--splits-dir", type=Path, default=Path("splits"))
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    parser.add_argument("--calibration-dir", type=Path, default=Path("results/calibration"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/evaluation"))
    parser.add_argument("--category", choices=["all"] + CATEGORIES, default="all")
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--cpu-threads", type=int, default=4)
    args = parser.parse_args()
    if min(args.batch_size, args.cpu_threads) < 1:
        parser.error("Batch size and CPU threads must be positive")
    torch.set_num_threads(args.cpu_threads)
    device = select_device(args.device)
    categories = CATEGORIES if args.category == "all" else [args.category]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    charts = args.output_dir / "charts"
    charts.mkdir(exist_ok=True)
    extractor = None
    summary = []
    print("Evaluating official test splits with fixed validation thresholds.", flush=True)

    for category in categories:
        model_path = args.models_dir / f"{category}_padim.pt"
        thresholds = json.loads((args.calibration_dir / f"{category}_thresholds.json").read_text())
        if thresholds["category"] != category or sha256(model_path) != thresholds["model_sha256"]:
            raise ValueError(f"Calibration/model mismatch: {category}")
        if thresholds["image_score"] != "maximum_smoothed_pixel_score" or thresholds["comparison"] != "strictly_greater":
            raise ValueError("Unsupported calibration scoring settings")
        if extractor is None:
            config = torch.load(model_path, map_location="cpu", weights_only=True)
            extractor = PaDiMFeatureExtractor(config["n_features"], config["seed"]).to(device).eval()
            del config
        predictor = PaDiMPredictor(model_path, device=device, extractor=extractor,
                                    sigma=thresholds["gaussian_sigma"])
        if predictor.image_size != thresholds["image_size"]:
            raise ValueError("Calibration/image-size mismatch")
        dataset = make_dataset(args.data_root, args.splits_dir, category, "test",
                               image_size=predictor.image_size, augment=False)
        loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=0, shuffle=False)
        all_scores, all_labels, pixel_scores, pixel_labels = [], [], [], []
        print(f"\n{category}: scoring {len(dataset)} test images...", flush=True)
        for batch in loader:
            scores, maps = predictor.predict(batch["image"])
            all_scores.append(scores)
            all_labels.append(batch["label"].numpy())
            pixel_scores.append(maps.reshape(-1))
            pixel_labels.append(batch["mask"][:, 0].numpy().reshape(-1).astype(np.uint8))
        scores = np.concatenate(all_scores)
        labels = np.concatenate(all_labels)
        pixels = np.concatenate(pixel_scores)
        masks = np.concatenate(pixel_labels)
        # Drop list references before sorting the full pixel vectors for AUROC/AP.
        del all_scores, all_labels, pixel_scores, pixel_labels
        image = binary_metrics(labels, scores, thresholds["image_threshold"])
        pixel = binary_metrics(masks, pixels, thresholds["pixel_threshold"])
        denominator = pixel["tp"] + pixel["fp"] + pixel["fn"]
        pixel["iou"] = pixel["tp"] / denominator if denominator else 0.0
        pixel["dice"] = pixel["f1"]
        result = {"category": category, "test_images": len(dataset),
                  "image_threshold": thresholds["image_threshold"],
                  "pixel_threshold": thresholds["pixel_threshold"],
                  "image": image, "pixel": pixel,
                  "test_manifest_sha256": sha256(args.splits_dir / f"{category}_test.csv"),
                  "model_sha256": thresholds["model_sha256"],
                  "pixel_evaluation_resolution": predictor.image_size,
                  "note": "Pixel metrics include all test images at the resized resolution; masks use nearest-neighbor resize."}
        (args.output_dir / f"{category}_metrics.json").write_text(json.dumps(result, indent=2) + "\n")
        with (args.output_dir / f"{category}_predictions.csv").open("w", newline="", encoding="utf-8") as file:
            fields = ["image_path", "defect_type", "true_label", "predicted_label", "anomaly_score", "image_threshold"]
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()
            for row, label, score in zip(dataset.rows, labels, scores):
                writer.writerow(dict(image_path=row["image_path"], defect_type=row["defect_type"],
                                     true_label=int(label), predicted_label=int(score > thresholds["image_threshold"]),
                                     anomaly_score=float(score), image_threshold=thresholds["image_threshold"]))
        plot_curves(category, labels, scores, image, charts)
        summary.append(dict(category=category, test_images=len(dataset),
                            image_auroc=image["auroc"], image_ap=image["average_precision"],
                            image_accuracy=image["accuracy"], image_precision=image["precision"],
                            image_recall=image["recall"], image_f1=image["f1"],
                            pixel_auroc=pixel["auroc"], pixel_ap=pixel["average_precision"],
                            pixel_f1=pixel["f1"], pixel_iou=pixel["iou"]))
        print(f"  Image AUROC={image['auroc']:.4f}; accuracy={image['accuracy']:.4f}; F1={image['f1']:.4f}", flush=True)
        print(f"  Pixel AUROC={pixel['auroc']:.4f}; Dice={pixel['dice']:.4f}; IoU={pixel['iou']:.4f}", flush=True)
        del predictor, pixels, masks, scores, labels

    with (args.output_dir / "metrics_summary.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    keys = [key for key in summary[0] if key not in ("category", "test_images")]
    macro = {key: float(np.mean([row[key] for row in summary])) for key in keys}
    (args.output_dir / "macro_metrics.json").write_text(json.dumps(
        {"categories": len(categories), "averaging": "equal-weight per-category mean", "metrics": macro}, indent=2) + "\n")
    fig, ax = plt.subplots(figsize=(11, 4.5))
    bars = ax.bar([row["category"] for row in summary], [100 * row["image_accuracy"] for row in summary])
    ax.bar_label(bars, fmt="%.1f", fontsize=8)
    ax.set(ylabel="Image classification accuracy (%)", ylim=(0, 110),
           title="PaDiM test accuracy by category — fixed validation thresholds")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(charts / "accuracy_by_category.png", dpi=160)
    plt.close(fig)
    print(f"\nEvaluation complete for {len(categories)} categories: {args.output_dir}", flush=True)
    print(f"Macro image AUROC={macro['image_auroc']:.4f}; macro pixel AUROC={macro['pixel_auroc']:.4f}", flush=True)


if __name__ == "__main__":
    main()
