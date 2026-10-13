"""Load PaDiM CSV manifests and preprocess MVTec images and defect masks.

Images: RGB, bilinear resize, float tensor, ImageNet normalization.
Masks: grayscale, nearest-neighbor resize, binary float tensor.
No augmentation is applied. Original dataset files are never modified.
"""
import argparse
import csv
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

CATEGORIES = ["bottle", "cable", "capsule", "carpet", "grid", "hazelnut",
              "leather", "metal_nut", "pill", "screw", "tile", "toothbrush",
              "transistor", "wood", "zipper"]
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class MVTecDataset(Dataset):
    """Return image [3,H,W], mask [1,H,W], label (0/1), and metadata."""

    def __init__(self, data_root, manifest_path, image_size=224):
        self.data_root = Path(data_root).expanduser().resolve()
        self.manifest_path = Path(manifest_path)
        if image_size < 1:
            raise ValueError("image_size must be positive")
        self.image_size = image_size
        with self.manifest_path.open(newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            required = {"category", "split", "image_path", "label", "defect_type", "mask_path"}
            if not required.issubset(reader.fieldnames or []):
                raise ValueError(f"Missing CSV columns: {self.manifest_path}")
            self.rows = list(reader)
        if not self.rows:
            raise ValueError(f"Empty manifest: {self.manifest_path}")
        for row in self.rows:
            label = int(row["label"])
            if label not in (0, 1):
                raise ValueError(f"Invalid label: {row}")
            if row["split"] in ("train", "validation") and label != 0:
                raise ValueError("Training and validation must contain only normal images")
            self._resolve(row["image_path"])
            if label == 1:
                if not row["mask_path"]:
                    raise ValueError(f"Anomaly has no mask: {row['image_path']}")
                self._resolve(row["mask_path"])
        self.image_transform = transforms.Compose([
            transforms.Resize((image_size, image_size),
                              interpolation=transforms.InterpolationMode.BILINEAR,
                              antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

    def _resolve(self, relative_path):
        path = (self.data_root / relative_path).resolve()
        if not path.is_relative_to(self.data_root):
            raise ValueError(f"Path outside data root: {relative_path}")
        if not path.is_file():
            raise FileNotFoundError(path)
        return path

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        label = int(row["label"])
        with Image.open(self._resolve(row["image_path"])) as source:
            original_size = source.size  # Width, height.
            image = self.image_transform(source.convert("RGB"))
        if label:
            with Image.open(self._resolve(row["mask_path"])) as source:
                if source.size != original_size:
                    raise ValueError(f"Image-mask size mismatch: {row['image_path']}")
                resized = source.convert("L").resize(
                    (self.image_size, self.image_size), Image.Resampling.NEAREST)
                mask = (transforms.functional.pil_to_tensor(resized) > 0).float()
        else:
            mask = torch.zeros((1, self.image_size, self.image_size), dtype=torch.float32)
        return {"image": image, "mask": mask, "label": label,
                "category": row["category"], "defect_type": row["defect_type"],
                "image_path": row["image_path"],
                "original_size": torch.tensor(original_size)}


def main():
    parser = argparse.ArgumentParser(description="Check preprocessing batches across MVTec splits")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--splits-dir", type=Path, default=Path("splits"))
    parser.add_argument("--category", choices=["all"] + CATEGORIES, default="all")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    categories = CATEGORIES if args.category == "all" else [args.category]
    checked = 0
    for category in categories:
        for split in ["train", "validation", "test"]:
            dataset = MVTecDataset(args.data_root,
                                   args.splits_dir / f"{category}_{split}.csv",
                                   args.image_size)
            loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
            batch = next(iter(loader))
            count = min(args.batch_size, len(dataset))
            assert batch["image"].shape == (count, 3, args.image_size, args.image_size)
            assert batch["mask"].shape == (count, 1, args.image_size, args.image_size)
            assert torch.isfinite(batch["image"]).all(), "Non-finite image values"
            assert ((batch["mask"] == 0) | (batch["mask"] == 1)).all(), "Non-binary mask"
            # Also check one anomalous test sample explicitly.
            if split == "test":
                anomaly_index = next((i for i, row in enumerate(dataset.rows)
                                      if int(row["label"]) == 1), None)
                if anomaly_index is None:
                    raise ValueError(f"No anomalous test sample: {category}")
                sample = dataset[anomaly_index]
                assert sample["mask"].shape == (1, args.image_size, args.image_size)
                assert ((sample["mask"] == 0) | (sample["mask"] == 1)).all()
            checked += 1
        print(f"{category:12} train / validation / test preprocessing: OK")
    print(f"\nPassed: {checked} split batch checks at {args.image_size} x {args.image_size}.")
    print("All manifest file paths checked; sampled images and masks transformed.")
    print("Original files unchanged. No augmentation applied.")


if __name__ == "__main__":
    main()
