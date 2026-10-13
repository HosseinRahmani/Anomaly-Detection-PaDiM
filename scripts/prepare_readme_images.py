"""Copy one existing defect visualization per category for the README gallery.

Run from the PADIM project root. Requires only Python's standard library.
Original visualization files are left unchanged.
"""
import argparse
import csv
import shutil
from pathlib import Path

CATEGORIES = (
    "bottle", "cable", "capsule", "carpet", "grid", "hazelnut",
    "leather", "metal_nut", "pill", "screw", "tile", "toothbrush",
    "transistor", "wood", "zipper",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--visualizations-dir", type=Path,
                        default=Path("results/visualizations"))
    parser.add_argument("--output-dir", type=Path, default=Path("docs/examples"))
    args = parser.parse_args()
    selected = []
    missing = []
    for category in CATEGORIES:
        images = sorted((args.visualizations_dir / category).glob("*.png"))
        defects = [p for p in images if not p.name.startswith("good_")]
        if not defects:
            missing.append(category)
        else:
            selected.append((category, defects[0]))
    # Validate all sources before changing any gallery files.
    if missing:
        raise SystemExit("No defect visualization found for: " + ", ".join(missing)
                         + ". Check --visualizations-dir. No files copied.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for category, source in selected:
        target = args.output_dir / f"{category}.png"
        if source.resolve() != target.resolve():
            shutil.copy2(source, target)
        print(f"{category}: {source.as_posix()} -> {target.as_posix()}")
    with (args.output_dir / "selection.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["category", "source_visualization", "readme_image"])
        for category, source in selected:
            writer.writerow([category, source.as_posix(),
                             (args.output_dir / f"{category}.png").as_posix()])
    print("Ready: 15 README images. Original results unchanged.")


if __name__ == "__main__":
    main()
