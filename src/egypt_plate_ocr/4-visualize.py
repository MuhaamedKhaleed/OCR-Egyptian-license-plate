"""Draw a plate result: one outer plate box, one inner box per detected character.

Every rectangle is a model detection from `predict_plates(..., include_boxes=True)`.
Nothing here estimates or invents a box.
"""

from pathlib import Path
from typing import Any

_PALETTE = [
    (230, 30, 30),
    (30, 90, 230),
    (0, 150, 60),
    (220, 130, 0),
    (150, 0, 180),
    (0, 160, 170),
    (120, 120, 0),
]
_ARABIC_FONTS = (
    "NotoSansArabic-Regular.ttf",
    "NotoNaskhArabic-Regular.ttf",
    "DejaVuSans.ttf",
    "arial.ttf",
    "Tahoma.ttf",
)


def _font(size: int) -> Any:
    from PIL import ImageFont

    for name in _ARABIC_FONTS:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def annotate_image(image: Path, plates: list[dict[str, Any]], output: Path) -> Path:
    """Save `image` with the plate box (black) and each character box (coloured).

    Character boxes are stored relative to the plate crop, so the crop origin is
    added back. The label under each box is the glyph the model read. If the
    default font cannot show Arabic, the class name is drawn instead.
    """
    from PIL import Image, ImageDraw

    if not image.is_file():
        raise ValueError(f"Image not found: {image}.")
    if output.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
        raise ValueError("Annotated output must end in .jpg, .jpeg or .png.")
    canvas = Image.open(image).convert("RGB")
    draw = ImageDraw.Draw(canvas)
    line = max(2, canvas.width // 400)
    for plate in plates:
        left, top, right, bottom = plate["box"]
        font = _font(max(12, (bottom - top) // 3))
        draw.rectangle((left, top, right, bottom), outline=(0, 0, 0), width=line + 1)
        for index, char in enumerate(plate.get("characters", [])):
            x1, y1, x2, y2 = char["box"]
            color = _PALETTE[index % len(_PALETTE)]
            draw.rectangle((left + x1, top + y1, left + x2, top + y2), outline=color, width=line)
            text = char["char"] if font.getmask(char["char"]).getbbox() else char["label"]
            draw.text(
                (left + (x1 + x2) / 2, bottom + 4 + (index % 2) * (font.size + 4)),
                text,
                font=font,
                fill=color,
                anchor="mt",
            )
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)
    return output
