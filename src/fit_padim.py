"""Fit per-category PaDiM Gaussian statistics from normal training images.

Place beside padim_features.py, augmented_dataset.py, and mvtec_dataset.py.
MPS/CUDA extracts features; CPU float64 computes regularized covariance.
Validation and test images are never used for fitting.
"""
import argparse
import hashlib
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from augmented_dataset import make_dataset
from mvtec_dataset import CATEGORIES, IMAGENET_MEAN, IMAGENET_STD
from padim_features import PaDiMFeatureExtractor, select_device


def fit_statistics(features, regularization, patch_chunk):
    # features: [N,D,H,W]. Work on small patch groups to bound memory usage.
    n, d, h, w = features.shape
    if n < 2:
        raise ValueError("At least two normal feature samples required")
    patches = features.flatten(2).permute(2, 0, 1)  # [H*W,N,D]
    means = torch.empty((h * w, d), dtype=torch.float32)
    precisions = torch.empty((h * w, d, d), dtype=torch.float32)
    identity = torch.eye(d, dtype=torch.float64)
    for start in range(0, h * w, patch_chunk):
        end = min(start + patch_chunk, h * w)
        x = patches[start:end].double()
        mean = x.mean(dim=1)
        centered = x - mean[:, None, :]
        covariance = centered.transpose(1, 2).bmm(centered) / (n - 1)
        covariance += regularization * identity
        cholesky = torch.linalg.cholesky(covariance)
        precision = torch.cholesky_inverse(cholesky)
        means[start:end] = mean.float()
        precisions[start:end] = precision.float()
    if not torch.isfinite(means).all() or not torch.isfinite(precisions).all():
        raise ValueError("Non-finite Gaussian statistics")
    return means, precisions, (h, w)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--splits-dir", type=Path, default=Path("splits"))
    parser.add_argument("--output-dir", type=Path, default=Path("models"))
    parser.add_argument("--category", choices=["all"] + CATEGORIES, default="all")
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--n-features", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--augmentation-passes", type=int, default=1,
                        help="Visits per training image; each visit samples new augmentation")
    parser.add_argument("--no-augmentation", action="store_true")
    parser.add_argument("--regularization", type=float, default=0.01)
    parser.add_argument("--patch-chunk", type=int, default=64)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if min(args.batch_size, args.augmentation_passes, args.patch_chunk, args.cpu_threads) < 1:
        parser.error("Batch size, passes, chunk size, and CPU threads must be positive")
    if args.regularization <= 0:
        parser.error("--regularization must be positive")
    torch.set_num_threads(args.cpu_threads)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = select_device(args.device)
    extractor = PaDiMFeatureExtractor(args.n_features, args.seed).to(device).eval()
    categories = CATEGORIES if args.category == "all" else [args.category]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Feature device: {device}; Gaussian fitting: CPU", flush=True)

    for category in categories:
        target = args.output_dir / f"{category}_padim.pt"
        if target.exists() and not args.overwrite:
            raise SystemExit(f"Already exists: {target}. Use another output directory or --overwrite.")
        # Separate category RNGs make all-category and single-category runs consistent.
        category_seed = args.seed + CATEGORIES.index(category)
        torch.manual_seed(category_seed)
        dataset = make_dataset(args.data_root, args.splits_dir, category, "train",
                               augment=not args.no_augmentation)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
        # Allocate one buffer instead of keeping duplicated feature batches.
        count = len(dataset) * args.augmentation_passes
        features = None
        offset = 0
        print(f"\n{category}: {len(dataset)} training images, {args.augmentation_passes} pass(es)", flush=True)
        for visit in range(args.augmentation_passes):
            for batch in loader:
                embedding = extractor(batch["image"].to(device)).cpu()
                if features is None:
                    features = torch.empty((count, *embedding.shape[1:]), dtype=torch.float32)
                features[offset:offset + len(embedding)] = embedding
                offset += len(embedding)
            print(f"  Extracted pass {visit + 1}/{args.augmentation_passes}", flush=True)
        assert offset == count
        print("  Fitting patch distributions...", flush=True)
        mean, precision, spatial_shape = fit_statistics(features, args.regularization, args.patch_chunk)
        del features, embedding
        manifest = args.splits_dir / f"{category}_train.csv"
        checkpoint = {
            "format_version": 1, "category": category,
            "backbone": "resnet18", "weights": "IMAGENET1K_V1",
            "layers": ["layer1", "layer2", "layer3"],
            "feature_indices": extractor.feature_indices.detach().cpu(),
            "mean": mean, "precision": precision,
            "spatial_shape": list(spatial_shape), "image_size": 224,
            "normalization_mean": list(IMAGENET_MEAN),
            "normalization_std": list(IMAGENET_STD),
            "image_interpolation": "bilinear", "feature_interpolation": "nearest",
            "n_features": args.n_features, "seed": args.seed,
            "augmentation_seed": category_seed,
            "augmentation": "none" if args.no_augmentation else "brightness_contrast_0.1",
            "augmentation_passes": args.augmentation_passes,
            "training_images": len(dataset), "fitted_samples": count,
            "regularization": args.regularization,
            "training_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
            "torch_version": str(torch.__version__),
        }
        temporary = target.with_suffix(".pt.tmp")
        torch.save(checkpoint, temporary)
        temporary.replace(target)
        print(f"  Saved: {target} ({target.stat().st_size / 1024**2:.1f} MiB)", flush=True)
        del mean, precision, checkpoint
    print("\nPaDiM fitting complete. Threshold calibration and evaluation remain.", flush=True)


if __name__ == "__main__":
    main()
