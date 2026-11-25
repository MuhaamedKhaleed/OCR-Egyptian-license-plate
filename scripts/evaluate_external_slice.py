"""Evaluate the packaged decoder on an external plate-image slice.

Runs the original recovered checkpoint through the package's own
``predict_plate``/``predict_plates`` paths on every image in a directory whose
names appear in a ground-truth CSV, then reports exact full-text, digit
sequence and localization counts per image and in total.

This is a convenience-sample diagnostic, not a general accuracy estimate:
source images may overlap the original training corpus, ground truth comes
from the source repository's own labels (visually reviewed per run), and any
slice used during development stops being held-out evidence.

Example:

    python scripts/evaluate_external_slice.py \
        --images external/Dataset \
        --ground-truth external/Results/ocr_results.csv \
        --weights /path/to/best.pt \
        --output /tmp/slice-results.json

Weights and image transcripts stay local; nothing is uploaded or committed.
"""

import argparse
import csv
import json
from pathlib import Path

GT_LETTER_MAP = {
    "ra": "ر",
    "noon": "ن",
    "ain": "ع",
    "alef": "أ",
    "seen": "س",
    "jeem": "ج",
    "lam": "ل",
    "fa": "ف",
    "ta": "ط",
    "dal": "د",
    "ha": "ه",
    "ba": "ب",
    "waw": "و",
    "sad": "ص",
    "qaf": "ق",
    "ya": "ي",
    "kaf": "ك",
    "meem": "م",
}
EASTERN_DIGITS = dict(zip("0123456789", "٠١٢٣٤٥٦٧٨٩", strict=True))


def parse_ground_truth(raw: str) -> tuple[str, str]:
    """Convert the source "ra seen waw | 0 4 9 6" style label to package form."""
    letters_raw, digits_raw = raw.split("|")
    letters = [GT_LETTER_MAP[token] for token in letters_raw.split()]
    digits = [EASTERN_DIGITS[token] for token in digits_raw.split()]
    return " ".join(letters), " ".join(digits)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, required=True, help="Directory of plate photos.")
    parser.add_argument("--ground-truth", type=Path, required=True, help="Source CSV with labels.")
    parser.add_argument("--weights", type=Path, required=True, help="Trusted local checkpoint.")
    parser.add_argument("--output", type=Path, required=True, help="Where to write JSON results.")
    parser.add_argument("--confidence", type=float, default=0.2)
    args = parser.parse_args()

    from egypt_plate_ocr.inference import predict_plate, predict_plates

    from ultralytics import YOLO

    model = YOLO(str(args.weights.resolve()))

    def factory(_path: str) -> YOLO:
        return model

    rows = {row["Image"]: row for row in csv.DictReader(args.ground_truth.open())}
    results = {}
    for name in sorted(rows):
        image = args.images / name
        if not image.is_file():
            continue
        full = predict_plate(image, args.weights, args.confidence, model_factory=factory)
        plates = predict_plates(image, args.weights, args.confidence, model_factory=factory)
        primary = max(plates, key=lambda p: p["confidence"]) if plates else None
        gt_letters, gt_digits = parse_ground_truth(rows[name]["GroundTruth"])
        results[name] = {
            "gt_letters": gt_letters,
            "gt_digits": gt_digits,
            "full_frame": full,
            "plates": [
                {key: plate[key] for key in ("box", "confidence", "letters", "numbers")}
                for plate in plates
            ],
            "full_exact": full["letters"] == gt_letters and full["numbers"] == gt_digits,
            "crop_exact": bool(primary)
            and primary["letters"] == gt_letters
            and primary["numbers"] == gt_digits,
            "crop_digits_ok": bool(primary) and primary["numbers"] == gt_digits,
            "localized": bool(plates),
        }
    summary = {
        "n": len(results),
        "full_exact": sum(item["full_exact"] for item in results.values()),
        "crop_exact": sum(item["crop_exact"] for item in results.values()),
        "crop_digits_ok": sum(item["crop_digits_ok"] for item in results.values()),
        "localized": sum(item["localized"] for item in results.values()),
    }
    args.output.write_text(json.dumps({"summary": summary, "images": results}, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
