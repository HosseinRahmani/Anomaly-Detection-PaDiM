"""On-the-fly training augmentation for the PaDiM MVTec pipeline.

Keep this file beside mvtec_dataset.py. Validation/test use the original
deterministic preprocessing. No augmented image files are written.
"""
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from mvtec_dataset import CATEGORIES, IMAGENET_MEAN, IMAGENET_STD, MVTecDataset


class AugmentedMVTecDataset(MVTecDataset):
    """Apply mild color jitter on each access to a normal training image."""

    def __init__(self, data_root, manifest_path, image_size=224):
        super().__init__(data_root, manifest_path, image_size)
        if any(row["split"] != "train" or int(row["label"]) != 0 for row in self.rows):
            raise ValueError("Augmentation is only allowed on normal training images")
        self.image_transform = transforms.Compose([
            transforms.Resize((image_size, image_size),
                              interpolation=transforms.InterpolationMode.BILINEAR,
                              antialias=True),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])


def make_dataset(data_root, splits_dir, category, split, image_size=224, augment=True):
    """Use this factory in training, calibration, and evaluation scripts."""
    if category not in CATEGORIES:
        raise ValueError(f"Unknown category: {category}")
    if split not in ("train", "validation", "test"):
        raise ValueError(f"Unknown split: {split}")
    dataset_class = AugmentedMVTecDataset if augment and split == "train" else MVTecDataset
    return dataset_class(data_root, Path(splits_dir) / f"{category}_{split}.csv", image_size)


def main():
    parser = argparse.ArgumentParser(description="Check on-the-fly training augmentation")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--splits-dir", type=Path, default=Path("splits"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    for category in CATEGORIES:
        for split in ("train", "validation", "test"):
            dataset = make_dataset(args.data_root, args.splits_dir, category, split)
            batch = next(iter(DataLoader(dataset, batch_size=4, num_workers=0)))
            assert torch.isfinite(batch["image"]).all()
            assert ((batch["mask"] == 0) | (batch["mask"] == 1)).all()
            if split == "train":
                assert isinstance(dataset, AugmentedMVTecDataset)
                # Resetting the seed must reproduce the same augmentation.
                torch.manual_seed(args.seed)
                first = dataset[0]["image"]
                torch.manual_seed(args.seed)
                second = dataset[0]["image"]
                assert torch.equal(first, second), "Seeded augmentation not reproducible"
            else:
                assert type(dataset) is MVTecDataset
                assert torch.equal(dataset[0]["image"], dataset[0]["image"])
        print(f"{category:12} training augmentation: OK; validation/test unchanged: OK")
    print("\nAll 15 categories passed. Brightness/contrast factors: 0.9 to 1.1.")
    print("No rotations, flips, or crops. No files written; dataset length unchanged.")


if __name__ == "__main__":
    main()
