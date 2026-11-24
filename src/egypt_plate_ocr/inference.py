"""Explicit local-weight inference. Importing this module does not load a model."""

from pathlib import Path
from typing import Any

from .core import ARABIC_MAPPING, Detection, decode_characters


def _overlap(first: list[float], second: list[float]) -> float:
    """Intersection over union for two xyxy character boxes."""
    x1, y1 = max(first[0], second[0]), max(first[1], second[1])
    x2, y2 = min(first[2], second[2]), min(first[3], second[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    area += max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = area - intersection
    return intersection / union if union > 0 else 0.0


def _unique_characters(candidates: list[tuple[str, list[float], float]]) -> list[Detection]:
    """Keep the stronger character when two classes claim the same box.

    The high IoU threshold only removes near-identical boxes. It does not merge
    adjacent letters, normalize Arabic spelling or group multiple plates.
    """
    kept: list[tuple[str, list[float], float]] = []
    for candidate in sorted(candidates, key=lambda item: item[2], reverse=True):
        if not any(_overlap(candidate[1], other[1]) >= 0.8 for other in kept):
            kept.append(candidate)
    return [Detection(label, (coords[0] + coords[2]) / 2) for label, coords, _ in kept]


def predict_plate(
    image: Path, weights: Path, confidence: float = 0.2, *, model_factory: Any = None
) -> dict[str, str]:
    """Predict on the requested image, never reuse a previous image's result.

    model_factory exists for offline unit testing. Real inference requires the
    optional Ultralytics dependency and compatible, trusted local model weights.
    """
    if not 0 <= confidence <= 1:
        raise ValueError("Confidence must be between 0 and 1.")
    if not image.is_file():
        raise ValueError(f"Image not found: {image}. Supply an existing image file.")
    if not weights.is_file():
        raise ValueError(f"Weights not found: {weights}. Supply trusted local trained weights.")
    if model_factory is None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ValueError('Install the inference extra: pip install -e ".[inference]"') from exc
        model_factory = YOLO
    model = model_factory(str(weights.resolve()))
    results = model.predict(source=str(image.resolve()), conf=confidence, verbose=False)
    if not results:
        return {"letters": "", "numbers": ""}
    result = results[0]
    candidates = []
    for box in result.boxes:
        class_id = int(box.cls.item())
        names = result.names
        label = names.get(class_id, "") if isinstance(names, dict) else names[class_id]
        if label not in ARABIC_MAPPING:
            continue
        coords = [float(value) for value in box.xyxy[0].tolist()]
        candidates.append((label, coords, float(box.conf.item())))
    return decode_characters(_unique_characters(candidates))


def predict_plates(
    image: Path, weights: Path, confidence: float = 0.2, *, model_factory: Any = None
) -> list[dict[str, Any]]:
    """Localize plates, then decode each crop independently.

    Returns no entries when no plate is detected. It never mixes characters
    across plates or silently falls back to whole-frame decoding.
    """
    if not 0 <= confidence <= 1:
        raise ValueError("Confidence must be between 0 and 1.")
    if not image.is_file() or not weights.is_file():
        raise ValueError("Image and trusted weights must both be existing files.")
    if model_factory is None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ValueError('Install the inference extra: pip install -e ".[inference]"') from exc
        model_factory = YOLO
    model = model_factory(str(weights.resolve()))
    frames = model.predict(source=str(image.resolve()), conf=confidence, verbose=False)
    if not frames:
        return []
    frame = frames[0]
    height, width = frame.orig_img.shape[:2]
    output = []
    for box in frame.boxes:
        if frame.names[int(box.cls.item())] != "License Plate":
            continue
        x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
        bounds = [
            max(0, int(x1) - 3),
            max(0, int(y1) - 3),
            min(width, int(x2) + 4),
            min(height, int(y2) + 4),
        ]
        left, top, right, bottom = bounds
        if right <= left or bottom <= top:
            continue
        crop = frame.orig_img[top:bottom, left:right]
        results = model.predict(source=crop, conf=confidence, verbose=False)
        candidates = []
        if results:
            result = results[0]
            for char in result.boxes:
                label = result.names[int(char.cls.item())]
                if label in ARABIC_MAPPING:
                    coords = [float(v) for v in char.xyxy[0].tolist()]
                    candidates.append((label, coords, float(char.conf.item())))
        output.append(
            {
                "box": bounds,
                "confidence": float(box.conf.item()),
                **decode_characters(_unique_characters(candidates)),
            }
        )
    return output
