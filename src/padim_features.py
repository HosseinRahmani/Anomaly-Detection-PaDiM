"""Frozen ImageNet ResNet18 patch embeddings for PaDiM (not yet fitted)."""
import argparse
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.models import ResNet18_Weights, resnet18

from augmented_dataset import make_dataset
from mvtec_dataset import CATEGORIES


class PaDiMFeatureExtractor(nn.Module):
    """Combine layer1/2/3 and retain a reproducible subset of channels.

    For 224x224 inputs, output shape is [batch, n_features, 56, 56].
    Normalization is applied by MVTecDataset, not this module.
    """

    def __init__(self, n_features=100, seed=42):
        super().__init__()
        if not 1 <= n_features <= 448:
            raise ValueError("n_features must be between 1 and 448")
        self.backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.backbone.requires_grad_(False)
        self.backbone.eval()
        generator = torch.Generator().manual_seed(seed)
        self.register_buffer("feature_indices", torch.randperm(448, generator=generator)[:n_features])

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()  # Keep batch normalization frozen even during fitting.
        return self

    @torch.no_grad()
    def forward(self, images):
        self.backbone.eval()
        model = self.backbone
        x = model.maxpool(model.relu(model.bn1(model.conv1(images))))
        layer1 = model.layer1(x)
        layer2 = model.layer2(layer1)
        layer3 = model.layer3(layer2)
        size = layer1.shape[-2:]
        features = torch.cat([
            layer1,
            F.interpolate(layer2, size=size, mode="nearest"),
            F.interpolate(layer3, size=size, mode="nearest"),
        ], dim=1)
        return features.index_select(1, self.feature_indices)


def select_device(name="auto"):
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    return torch.device(name)


def main():
    parser = argparse.ArgumentParser(description="Check PaDiM's pretrained feature extractor")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--splits-dir", type=Path, default=Path("splits"))
    parser.add_argument("--category", choices=CATEGORIES, default="metal_nut")
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--n-features", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    torch.manual_seed(args.seed)
    device = select_device(args.device)
    dataset = make_dataset(args.data_root, args.splits_dir, args.category, "train", augment=True)
    batch = next(iter(DataLoader(dataset, batch_size=args.batch_size, num_workers=0)))
    extractor = PaDiMFeatureExtractor(args.n_features, args.seed).to(device).eval()
    images = batch["image"].to(device)
    embeddings = extractor(images)
    assert tuple(embeddings.shape) == (len(images), args.n_features, 56, 56)
    assert torch.isfinite(embeddings).all(), "Non-finite features"
    assert not embeddings.requires_grad
    assert not any(p.requires_grad for p in extractor.parameters())
    print(f"Device: {device}")
    print("Backbone: ImageNet-pretrained ResNet18, frozen")
    print(f"Input: {tuple(images.shape)}")
    print("Combined channels: 448 (64 + 128 + 256)")
    print(f"PaDiM patch embeddings: {tuple(embeddings.shape)}")
    print("Feature extraction: OK")
    print("Next: fit per-category Gaussian statistics; no anomaly model has been fitted yet.")


if __name__ == "__main__":
    main()
