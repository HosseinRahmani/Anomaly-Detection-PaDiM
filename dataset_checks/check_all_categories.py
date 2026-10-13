"""Read-only folder counts and image-mask pairing checks for all 15 categories."""
from pathlib import Path

root = Path("/Users/hossein/Desktop/02-Personal/07-Github/PADIM")
categories = ["bottle", "cable", "capsule", "carpet", "grid", "hazelnut",
              "leather", "metal_nut", "pill", "screw", "tile", "toothbrush",
              "transistor", "wood", "zipper"]
issues = []
for category in categories:
    base = root / category
    required = [base / "train/good", base / "test/good", base / "ground_truth"]
    if any(not p.is_dir() for p in required):
        issues.append(f"{category}: missing required folder")
        continue
    train = list((base / "train/good").glob("*.png"))
    test = list((base / "test").glob("*/*.png"))
    defects = [p for p in test if p.parent.name != "good"]
    if not train or not test or not defects:
        issues.append(f"{category}: empty training, test, or defect set")
    for img in defects:
        mask = base / "ground_truth" / img.parent.name / f"{img.stem}_mask.png"
        if not mask.is_file():
            issues.append(f"Missing mask: {mask}")
    for mask in (base / "ground_truth").glob("*/*.png"):
        stem = mask.stem
        if stem.endswith("_mask"):
            stem = stem[:-5]
        img = base / "test" / mask.parent.name / f"{stem}.png"
        if not img.is_file():
            issues.append(f"Unmatched mask: {mask}")
    print(f"{category:12} train={len(train):3} test={len(test):3} defective={len(defects):3}")

print(f"\nIssues found: {len(issues)}")
for issue in issues:
    print(issue)
