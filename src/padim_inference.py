"""Shared PaDiM scoring for calibration, evaluation, and visualization."""
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import gaussian_filter
from torch.nn import functional as F

from padim_features import PaDiMFeatureExtractor, select_device


class PaDiMPredictor:
    """Return raw anomaly maps and max-map image scores (not probabilities)."""

    def __init__(self, model_path, device="auto", extractor=None, sigma=4.0, patch_chunk=128):
        self.model_path = Path(model_path)
        self.checkpoint = torch.load(self.model_path, map_location="cpu", weights_only=True)
        checkpoint = self.checkpoint
        if checkpoint["format_version"] != 1:
            raise ValueError("Unsupported checkpoint format")
        if checkpoint["backbone"] != "resnet18" or checkpoint["weights"] != "IMAGENET1K_V1":
            raise ValueError("Unsupported backbone/weights")
        self.device = select_device(str(device))
        self.extractor = extractor or PaDiMFeatureExtractor(checkpoint["n_features"], checkpoint["seed"])
        self.extractor.to(self.device).eval()
        indices = checkpoint["feature_indices"].to(self.device)
        if self.extractor.feature_indices.shape != indices.shape:
            raise ValueError("Feature dimension mismatch")
        self.extractor.feature_indices.copy_(indices)
        self.mean = checkpoint["mean"]
        self.precision = checkpoint["precision"]
        self.spatial_shape = tuple(checkpoint["spatial_shape"])
        self.image_size = checkpoint["image_size"]
        self.sigma = sigma
        self.patch_chunk = patch_chunk
        if sigma < 0 or patch_chunk < 1:
            raise ValueError("Invalid smoothing or chunk size")

    @torch.no_grad()
    def predict(self, images):
        embeddings = self.extractor(images.to(self.device)).cpu()
        if tuple(embeddings.shape[-2:]) != self.spatial_shape:
            raise ValueError("Embedding spatial shape differs from fitted model")
        patches = embeddings.flatten(2).permute(0, 2, 1)  # [B,P,D]
        distances = torch.empty(patches.shape[:2], dtype=torch.float32)
        for start in range(0, patches.shape[1], self.patch_chunk):
            end = min(start + self.patch_chunk, patches.shape[1])
            delta = patches[:, start:end] - self.mean[start:end]
            squared = torch.einsum("bpd,pde,bpe->bp", delta, self.precision[start:end], delta)
            distances[:, start:end] = squared.clamp_min(0).sqrt()
        maps = distances.reshape(-1, 1, *self.spatial_shape)
        maps = F.interpolate(maps, size=(self.image_size, self.image_size),
                             mode="bilinear", align_corners=False)[:, 0].numpy()
        if self.sigma:
            maps = np.stack([gaussian_filter(m, sigma=self.sigma, mode="reflect") for m in maps])
        if not np.isfinite(maps).all():
            raise ValueError("Non-finite anomaly maps")
        scores = maps.max(axis=(1, 2))
        return scores, maps
