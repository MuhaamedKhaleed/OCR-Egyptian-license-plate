"""Thin Typer interface with JSON on stdout and errors on stderr."""

import json
import logging
from pathlib import Path

import typer
from click.exceptions import Exit
from typer.main import get_command

from .inference import predict_plate, predict_plates

app = typer.Typer(add_completion=False, pretty_exceptions_enable=False)
log = logging.getLogger(__name__)


@app.command()
def predict(
    image: Path = typer.Option(..., help="Local image of one plate crop."),
    weights: Path = typer.Option(..., help="Trusted compatible local YOLO weights."),
    confidence: float = typer.Option(0.2, help="Detection confidence between 0 and 1."),
    auto_crop: bool = typer.Option(False, help="Find and decode each plate independently."),
    detector_label: str = typer.Option(
        "License Plate", help="Plate class name inside the supplied weights."
    ),
    localize_only: bool = typer.Option(
        False, help="Return plate boxes only, without character decoding. Requires --auto-crop."
    ),
    char_boxes: bool = typer.Option(
        False,
        help="Add the model's per-character boxes to each plate. Requires --auto-crop.",
    ),
    annotate: Path | None = typer.Option(
        None,
        help="Write an image with the plate box and one box per character. Requires --auto-crop.",
    ),
) -> None:
    """Decode one local plate image. No weights are downloaded by this wrapper."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        if localize_only and not auto_crop:
            raise ValueError("--localize-only requires --auto-crop.")
        if (char_boxes or annotate) and not auto_crop:
            raise ValueError("--char-boxes and --annotate require --auto-crop.")
        if (char_boxes or annotate) and localize_only:
            raise ValueError("--char-boxes and --annotate cannot be used with --localize-only.")
        result = (
            predict_plates(
                image,
                weights,
                confidence,
                detector_label=detector_label,
                localize_only=localize_only,
                include_boxes=bool(char_boxes or annotate),
            )
            if auto_crop
            else predict_plate(image, weights, confidence)
        )
        if annotate:
            from .visualize import annotate_image

            annotate_image(image, result, annotate)
    except (ValueError, OSError, RuntimeError) as exc:
        log.error("%s", exc)
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(2) from exc
    typer.echo(json.dumps(result, ensure_ascii=False))


def main(argv: list[str] | None = None) -> int:
    """Run the CLI and return its exit status, including validation failures."""
    from click import ClickException

    try:
        result = get_command(app).main(args=argv, standalone_mode=False)
    except ClickException as exc:
        exc.show()
        return exc.exit_code
    except Exit as exc:
        return exc.exit_code
    return result if isinstance(result, int) else 0
