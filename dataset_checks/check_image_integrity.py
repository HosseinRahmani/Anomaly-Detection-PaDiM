"""Read-only PNG decoding and image-mask dimension check."""
from pathlib import Path
try:
    from PIL import Image
except ImportError:
    raise SystemExit("Run: python3 -m pip install Pillow\nThen rerun this check.")

root = Path("/Users/hossein/Desktop/02-Personal/07-Github/PADIM")
if not root.is_dir():
    raise SystemExit(f"Folder not found: {root}")
issues = []
checked = 0
for category in sorted(root.iterdir()):
    if not (category / "train").is_dir():
        continue
    files = []
    for folder in ["train", "test", "ground_truth"]:
        files.extend((category / folder).rglob("*.png"))
    for path in files:
        try:
            with Image.open(path) as image:
                image.load()
            checked += 1
        except Exception as error:
            issues.append(f"Unreadable: {path}: {error}")
    for path in (category / "test").glob("*/*.png"):
        if path.parent.name == "good":
            continue
        mask = category / "ground_truth" / path.parent.name / f"{path.stem}_mask.png"
        try:
            with Image.open(path) as image, Image.open(mask) as label:
                if image.size != label.size:
                    issues.append(f"Size mismatch: {path}")
        except Exception:
            issues.append(f"Could not check image-mask sizes: {path}")

print(f"Readable images and masks: {checked}")
print(f"Issues found: {len(issues)}")
for issue in issues:
    print(issue)
