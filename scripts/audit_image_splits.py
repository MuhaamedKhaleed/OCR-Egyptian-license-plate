"""Read-only duplicate screening, never an independent-test certification."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp", ".ppm"}


def image_fingerprints(path: Path) -> tuple[str, int]:
    """Hash EXIF-oriented RGB pixels and return a 256-bit horizontal dHash."""
    from PIL import Image, ImageOps

    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
        image.load()
        width, height = image.size
        digest = hashlib.sha256(f"RGB:{width}:{height}:".encode() + image.tobytes()).hexdigest()
        gray = image.convert("L").resize((17, 16), Image.Resampling.BILINEAR)
        pixels = list(gray.tobytes())
        dhash = 0
        for y in range(16):
            for x in range(16):
                dhash = (dhash << 1) | (pixels[y * 17 + x + 1] > pixels[y * 17 + x])
    return digest, dhash


def audit(
    splits: dict[str, Path],
    *,
    visual: bool = False,
    threshold: int = 16,
    max_comparisons: int = 5_000_000,
    max_findings: int = 10_000,
) -> dict:
    """Screen explicitly named directories; do not discover other/test folders.

    Every cross-split pair is compared unless the declared limit would be
    exceeded, in which case the audit fails before pair processing. Findings
    are counts plus a bounded sample. No files are removed or relabeled.
    """
    if len(splits) < 2 or any(not name or not path.is_dir() for name, path in splits.items()):
        raise ValueError("Supply at least two named existing split directories.")
    roots = [path.resolve() for path in splits.values()]
    if any(
        a == b or a in b.parents or b in a.parents
        for i, a in enumerate(roots)
        for b in roots[i + 1 :]
    ):
        raise ValueError("Split roots must be distinct and non-nested.")
    if not 0 <= threshold <= 256 or max_comparisons < 1 or max_findings < 1:
        raise ValueError("Invalid screening threshold or limits.")
    records = {}
    errors = []
    for name, root in splits.items():
        items = []
        for path in sorted(root.rglob("*")):
            if path.suffix.lower() not in SUFFIXES:
                continue
            relative = str(path.relative_to(root))
            if (
                path.is_symlink()
                or not path.is_file()
                or root.resolve() not in path.resolve().parents
            ):
                errors.append({"split": name, "path": relative, "error": "unsafe_path"})
                continue
            row = {"path": relative}
            try:
                row["bytes"] = hashlib.sha256(path.read_bytes()).hexdigest()
                if visual:
                    row["pixels"], row["dhash"] = image_fingerprints(path)
            except (OSError, ValueError) as exc:
                # No raw provider messages or image data in reports.
                errors.append({"split": name, "path": relative, "error": type(exc).__name__})
                continue
            items.append(row)
        if not items:
            raise ValueError(f"Split {name} has no readable image files.")
        records[name] = items
    names = list(records)
    comparisons = sum(
        len(records[a]) * len(records[b]) for i, a in enumerate(names) for b in names[i + 1 :]
    )
    if comparisons > max_comparisons:
        raise ValueError(
            "Cross-split comparison limit exceeded; explicitly raise it or narrow scope."
        )
    findings = []
    counts = {"same_bytes": 0, "same_pixels": 0, "perceptual_candidate": 0}
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            for a in records[left]:
                for b in records[right]:
                    distance = (a["dhash"] ^ b["dhash"]).bit_count() if visual else None
                    kind = (
                        "same_bytes"
                        if a["bytes"] == b["bytes"]
                        else "same_pixels"
                        if visual and a["pixels"] == b["pixels"]
                        else "perceptual_candidate"
                        if distance is not None and distance <= threshold
                        else None
                    )
                    if kind:
                        counts[kind] += 1
                        if len(findings) < max_findings:
                            findings.append(
                                {
                                    "kind": kind,
                                    "left_split": left,
                                    "left": a["path"],
                                    "right_split": right,
                                    "right": b["path"],
                                    "dhash_distance": distance,
                                }
                            )
    return {
        "images": {name: len(items) for name, items in records.items()},
        "cross_split_comparisons": comparisons,
        "visual": visual,
        "dhash_bits": 256 if visual else None,
        "threshold": threshold if visual else None,
        "counts": counts,
        "findings": findings,
        "findings_truncated": sum(counts.values()) > len(findings),
        "errors": errors,
        "complete_read": not errors,
        "limits": [
            "Screening only, not evidence of independent images, vehicles, authors or capture sessions.",
            "Perceptual candidates require visual review; false positives and missed duplicates are possible.",
            "No grouping, deletion, annotation parsing, label correctness or authenticity certification.",
            "Only explicitly named folders read. Inspecting heldout pixels exposes that source.",
            "Reports contain local file names; keep private if names identify people or registrations.",
            "A zero finding count does not prove no leakage. Byte-only mode misses recoding and transforms.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", action="append", required=True, metavar="NAME=DIR")
    parser.add_argument("--visual", action="store_true", help="Opt-in Pillow pixel/dHash read.")
    parser.add_argument("--threshold", type=int, default=16)
    parser.add_argument("--max-comparisons", type=int, default=5_000_000)
    parser.add_argument("--max-findings", type=int, default=10_000)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        if args.output.exists():
            raise ValueError("Output exists; choose a new file.")
        splits = {}
        for item in args.split:
            if "=" not in item:
                raise ValueError("Each split must be NAME=DIR.")
            name, path = item.split("=", 1)
            if name in splits:
                raise ValueError("Split names must be unique.")
            splits[name] = Path(path)
        report = audit(
            splits,
            visual=args.visual,
            threshold=args.threshold,
            max_comparisons=args.max_comparisons,
            max_findings=args.max_findings,
        )
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
    except (OSError, ValueError, ImportError) as exc:
        parser.exit(2, f"Error: {exc}\n")
    print(
        json.dumps({k: v for k, v in report.items() if k not in {"findings", "limits", "errors"}})
    )
    if report["errors"]:
        parser.exit(2, "Some image reads failed; inspect the saved error list.\n")


if __name__ == "__main__":
    main()
