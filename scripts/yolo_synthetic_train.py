"""Train a character-reading YOLO model from the synthetic plates, then read a plate.

Outsider route to a REAL output with no private weights and no private data:

    python scripts/synthetic_plate_cpu.py generate --data ./syn
    python scripts/fetch_detector.py --output ./detector
    python scripts/yolo_synthetic_train.py --data ./syn --base ./detector/egyptian-plate-detector.pt \
        --output ./char-model --epochs 30

The base .pt only supplies the small YOLOv8 architecture (weights are re-initialised,
``pretrained=False``). The result is a 26-class character detector (17 letters, digits
1-9) that ``egypt-plate-ocr --weights ./char-model/train/weights/best.pt`` can run.
The data is rendered with fonts, not photographed, so it will read synthetic plates far
better than real ones. It is a starting point to improve, not a benchmark.
All output directories must be new.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

# Glyph -> class name used by egypt_plate_ocr.core.ARABIC_MAPPING.
LETTER_CLASSES = {
    "أ": "alef", "ب": "baa", "ج": "geem", "د": "daal", "ر": "raa", "س": "seen",
    "ص": "sad", "ط": "taa", "ع": "aain", "ف": "faa", "ق": "qaf", "ل": "laam",
    "م": "meem", "ن": "noon", "ه": "haa", "و": "waaw", "ي": "yaa",
}  # fmt: skip
DIGIT_CLASSES = {"١٢٣٤٥٦٧٨٩"[i]: str(i + 1) for i in range(9)}
NAMES = [*LETTER_CLASSES.values(), *DIGIT_CLASSES.values()]
GLYPH_TO_ID = {g: NAMES.index(n) for g, n in {**LETTER_CLASSES, **DIGIT_CLASSES}.items()}


def export_yolo(data: Path, out: Path) -> Path:
    """Write images/labels in Ultralytics layout from the generator's manifest.json."""
    from PIL import Image

    rows = json.loads((data / "manifest.json").read_text(encoding="utf-8"))
    out.mkdir(parents=True, exist_ok=False)
    split_dir = {"train": "train", "dev": "val", "test": "test"}
    for row in rows:
        split = split_dir[row["split"]]
        img_dir, lbl_dir = out / "images" / split, out / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        src = data / row["file"]
        w, h = Image.open(src).size
        shutil.copy(src, img_dir / src.name)
        lines = []
        for glyph, (x1, y1, x2, y2) in zip(row["visual_left_to_right"], row["boxes"], strict=True):
            cx, cy, bw, bh = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h, (x2 - x1) / w, (y2 - y1) / h
            lines.append(f"{GLYPH_TO_ID[glyph]} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        (lbl_dir / f"{src.stem}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    names = "\n".join(f"  {i}: {n}" for i, n in enumerate(NAMES))
    yaml_path = out / "data.yaml"
    yaml_path.write_text(
        f"path: {out.resolve()}\ntrain: images/train\nval: images/val\nnames:\n{names}\n",
        encoding="utf-8",
    )
    return yaml_path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument(
        "--data", type=Path, required=True, help="output of synthetic_plate_cpu.py generate"
    )
    p.add_argument(
        "--base", type=Path, required=True, help="local .pt that supplies the architecture"
    )
    p.add_argument("--output", type=Path, required=True, help="new directory")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--image-size", type=int, default=352)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    if args.output.exists():
        p.error("Choose a new output directory.")
    from egypt_plate_ocr.training import train_local

    args.output.mkdir(parents=True)
    yaml_path = export_yolo(args.data, args.output / "yolo-data")
    train_local(
        yaml_path, args.base, args.output / "train", epochs=args.epochs,
        image_size=args.image_size, batch=args.batch, device=args.device,
    )  # fmt: skip
    best = args.output / "train" / "weights" / "best.pt"
    print(json.dumps({"weights": str(best), "exists": best.is_file(), "classes": len(NAMES)}))


if __name__ == "__main__":
    main()
