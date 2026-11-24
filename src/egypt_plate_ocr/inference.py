"""Explicit local-weight inference. Importing this module does not load a model."""

from pathlib import Path
from typing import Any

from .core import Detection, decode_characters


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
    detections = []
    for box in result.boxes:
        class_id = int(box.cls.item())
        names = result.names
        label = names.get(class_id, "") if isinstance(names, dict) else names[class_id]
        coords = box.xyxy[0].tolist()
        detections.append(Detection(label, (float(coords[0]) + float(coords[2])) / 2))
    return decode_characters(detections)
