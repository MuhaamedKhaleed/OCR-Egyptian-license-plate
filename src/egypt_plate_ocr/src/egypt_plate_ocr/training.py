"""Explicit local CPU training, with no remote checkpoint or credential fallback."""

from pathlib import Path
from typing import Any


def train_local(
    data: Path,
    weights: Path,
    output: Path,
    *,
    epochs: int = 1,
    image_size: int = 640,
    model_factory: Any = None,
) -> dict[str, str]:
    """Train only after local artifacts and a new destination are supplied.

    The dataset YAML and weights must be trusted. Dataset permissions, valid
    annotations and an independent evaluation set remain the caller's job.
    """
    if not data.is_file() or data.suffix.lower() not in {".yaml", ".yml"}:
        raise ValueError("Supply an existing trusted local dataset YAML.")
    if not weights.is_file() or weights.suffix.lower() != ".pt":
        raise ValueError("Supply existing trusted local .pt weights; no download fallback.")
    if output.exists():
        raise ValueError("Output already exists; choose a new training destination.")
    if epochs < 1 or image_size < 32 or image_size % 32:
        raise ValueError("Epochs must be positive and image size a positive multiple of 32.")
    import yaml  # type: ignore[import-untyped]

    settings = yaml.safe_load(data.read_text(encoding="utf-8"))
    if not isinstance(settings, dict) or "download" in settings:
        raise ValueError("Dataset YAML must be a mapping without download directives.")
    root_setting = settings.get("path", ".")
    if not isinstance(root_setting, str) or "://" in root_setting:
        raise ValueError("Dataset path must be a local directory.")
    root = Path(root_setting)
    if not root.is_absolute():
        root = data.resolve().parent / root
    directories = []
    for split in ("train", "val"):
        value = settings.get(split)
        if not isinstance(value, str) or "://" in value:
            raise ValueError("Train and val must each name a local image directory.")
        directory = Path(value)
        if not directory.is_absolute():
            directory = root / directory
        if not directory.is_dir():
            raise ValueError("Train and val image directories must exist locally.")
        directories.append(directory.resolve())
    if directories[0] == directories[1]:
        raise ValueError("Train and val must use different directories.")
    if model_factory is None:
        from ultralytics import YOLO

        model_factory = YOLO
    model = model_factory(str(weights.resolve()))
    model.train(
        data=str(data.resolve()),
        epochs=epochs,
        batch=1,
        imgsz=image_size,
        device="cpu",
        workers=0,
        cache=False,
        pretrained=False,
        amp=False,
        plots=False,
        project=str(output.resolve().parent),
        name=output.name,
        exist_ok=False,
    )
    return {"output": str(output.resolve()), "device": "cpu"}
