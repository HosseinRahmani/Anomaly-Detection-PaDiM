"""Calibrate PaDiM thresholds on normal validation images only."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from augmented_dataset import make_dataset
from mvtec_dataset import CATEGORIES
from padim_features import PaDiMFeatureExtractor, select_device
from padim_inference import PaDiMPredictor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--splits-dir", type=Path, default=Path("splits"))
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/calibration"))
    parser.add_argument("--category", choices=["all"] + CATEGORIES, default="all")
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--image-quantile", type=float, default=0.95)
    parser.add_argument("--pixel-quantile", type=float, default=0.995)
    parser.add_argument("--sigma", type=float, default=4.0)
    parser.add_argument("--cpu-threads", type=int, default=4)
    args = parser.parse_args()
    if not 0 < args.image_quantile < 1 or not 0 < args.pixel_quantile < 1:
        parser.error("Quantiles must be strictly between 0 and 1")
    if min(args.batch_size, args.cpu_threads) < 1 or args.sigma < 0:
        parser.error("Batch size/threads must be positive; sigma must be nonnegative")
    torch.set_num_threads(args.cpu_threads)
    device = select_device(args.device)
    categories = CATEGORIES if args.category == "all" else [args.category]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    extractor = None
    summary = {}
    print(f"Feature device: {device}; scoring: CPU. Validation only.", flush=True)
    for category in categories:
        model_path = args.models_dir / f"{category}_padim.pt"
        # Load/check each checkpoint and share the frozen CNN across categories.
        if extractor is None:
            config = torch.load(model_path, map_location="cpu", weights_only=True)
            extractor = PaDiMFeatureExtractor(config["n_features"], config["seed"]).to(device).eval()
            del config
        predictor = PaDiMPredictor(model_path, device=device, extractor=extractor, sigma=args.sigma)
        if predictor.checkpoint["category"] != category:
            raise ValueError(f"Checkpoint category mismatch: {model_path}")
        dataset = make_dataset(args.data_root, args.splits_dir, category, "validation",
                               image_size=predictor.image_size, augment=False)
        loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=0, shuffle=False)
        score_batches, map_batches = [], []
        print(f"\n{category}: scoring {len(dataset)} normal validation images...", flush=True)
        for batch in loader:
            if not (batch["label"] == 0).all():
                raise ValueError("Calibration requires normal images only")
            scores, maps = predictor.predict(batch["image"])
            score_batches.append(scores)
            map_batches.append(maps)
        scores = np.concatenate(score_batches)
        maps = np.concatenate(map_batches)
        image_threshold = float(np.quantile(scores, args.image_quantile))
        pixel_threshold = float(np.quantile(maps, args.pixel_quantile))
        manifest = args.splits_dir / f"{category}_validation.csv"
        # Bind thresholds to the exact fitted checkpoint and validation manifest.
        digest = hashlib.sha256()
        with model_path.open("rb") as file:
            for block in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(block)
        settings = {
            "format_version": 1, "category": category,
            "image_threshold": image_threshold, "pixel_threshold": pixel_threshold,
            "image_quantile": args.image_quantile, "pixel_quantile": args.pixel_quantile,
            "image_score": "maximum_smoothed_pixel_score", "comparison": "strictly_greater",
            "map_resize": "bilinear_align_corners_false",
            "gaussian_sigma": args.sigma, "gaussian_mode": "reflect",
            "score_type": "raw_mahalanobis_distance", "image_size": predictor.image_size,
            "validation_images": len(dataset),
            "validation_image_exceedance": float((scores > image_threshold).mean()),
            "validation_pixel_exceedance": float((maps > pixel_threshold).mean()),
            "validation_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
            "model_sha256": digest.hexdigest(),
            "note": "Normal-only empirical quantiles; future false-positive rates are not guaranteed.",
        }
        target = args.output_dir / f"{category}_thresholds.json"
        target.write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")
        np.savez_compressed(args.output_dir / f"{category}_validation_scores.npz",
                            image_scores=scores,
                            image_paths=np.array([row["image_path"] for row in dataset.rows]))
        summary[category] = settings
        print(f"  image threshold={image_threshold:.4f}; pixel threshold={pixel_threshold:.4f}", flush=True)
        print(f"  Validation images flagged: {int((scores > image_threshold).sum())}/{len(scores)}", flush=True)
        del predictor, maps, map_batches, score_batches
    (args.output_dir / "calibration_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"\nCalibration complete for {len(categories)} categories: {args.output_dir}", flush=True)
    print("Test sets unused. Use these same map/scoring settings during evaluation.", flush=True)


if __name__ == "__main__":
    main()
