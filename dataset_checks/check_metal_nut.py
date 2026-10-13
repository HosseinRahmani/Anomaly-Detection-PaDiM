"""Read-only folder counts and image-mask pairing check for metal_nut."""
from pathlib import Path

root = Path("/Users/hossein/Desktop/02-Personal/07-Github/PADIM/metal_nut")
if not root.is_dir():
    raise SystemExit(f"Folder not found: {root}")

for folder in ["train/good", "test/good", "test/bent", "test/color",
               "test/flip", "test/scratch"]:
    path = root / folder
    if not path.is_dir():
        raise SystemExit(f"Missing folder: {path}")
    print(f"{folder}: {len(list(path.glob('*.png')))} images")

missing, orphan = [], []
for defect in ["bent", "color", "flip", "scratch"]:
    masks_dir = root / "ground_truth" / defect
    if not masks_dir.is_dir():
        raise SystemExit(f"Missing folder: {masks_dir}")
    for img in (root / "test" / defect).glob("*.png"):
        mask = masks_dir / f"{img.stem}_mask.png"
        if not mask.is_file():
            missing.append(str(mask))
    for mask in masks_dir.glob("*.png"):
        stem = mask.stem
        if stem.endswith("_mask"):
            stem = stem[:-5]
        if not (root / "test" / defect / f"{stem}.png").is_file():
            orphan.append(str(mask))

print(f"\nMissing masks: {len(missing)}")
print(f"Masks without matching images: {len(orphan)}")
for path in missing + orphan:
    print(path)
