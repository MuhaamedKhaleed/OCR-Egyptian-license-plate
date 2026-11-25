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
    "/usr/share/fonts/truetype/noto/NotoSansArabic-Regular.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "NotoNaskhArabic-Regular.ttf",
    "DejaVuSans.ttf",
    "arial.ttf",
    "Tahoma.ttf",
)


def _font(size: int) -> tuple[Any, bool]:
    """Return (font, can_draw_arabic). Falls back to a scalable Latin font."""
    from PIL import ImageFont

    candidates = list(_ARABIC_FONTS)
    try:
        import matplotlib

        candidates.append(
            str(Path(matplotlib.get_data_path()) / "fonts" / "ttf" / "DejaVuSans.ttf")
        )
    except ImportError:
        pass
    for name in candidates:
        try:
            return ImageFont.truetype(name, size), True
        except OSError:
            continue
    try:
        return ImageFont.load_default(size), False
    except TypeError:
        return ImageFont.load_default(), False


def annotate_image(image: Path, plates: list[dict[str, Any]], output: Path) -> Path:
    """Save a zoomed view of each plate: boxes on the photo, every label below.

    The view is the plate region plus a margin, scaled up to at least 1400 px
    wide, then extended downward with a white strip. Each character label is drawn
    directly under its own box on one row, so no label can fall off the image, and
    the full recognized string is printed in a banner at the bottom. With several
    plates the views are stacked. Boxes are the model's own detections (character
    boxes are relative to the plate crop). If no Arabic-capable font is found the
    class name (for example qaf, 2) is drawn instead.
    """
    from PIL import Image

    if not image.is_file():
        raise ValueError(f"Image not found: {image}.")
    if output.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
        raise ValueError("Annotated output must end in .jpg, .jpeg or .png.")
    source = Image.open(image).convert("RGB")
    views = [_plate_view(source, plate) for plate in plates] or [source]
    width = max(view.width for view in views)
    sheet = Image.new("RGB", (width, sum(view.height for view in views)), (255, 255, 255))
    top = 0
    for view in views:
        sheet.paste(view, (0, top))
        top += view.height
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output)
    return output


def _plate_view(source: Any, plate: dict[str, Any]) -> Any:
    from PIL import Image, ImageDraw

    left, top, right, bottom = plate["box"]
    pad_x, pad_y = max(10, (right - left) // 8), max(10, (bottom - top) // 4)
    x0, y0 = max(0, left - pad_x), max(0, top - pad_y)
    x1, y1 = min(source.width, right + pad_x), min(source.height, bottom + pad_y)
    scale = max(1.0, 1400 / (x1 - x0))
    view = source.crop((x0, y0, x1, y1)).resize(
        (round((x1 - x0) * scale), round((y1 - y0) * scale)), Image.Resampling.LANCZOS
    )

    def to_view(x: float, y: float) -> tuple[float, float]:
        return (x - x0) * scale, (y - y0) * scale

    size = max(28, view.width // 22)
    font, arabic_ok = _font(size)
    chars = plate.get("characters", [])
    text = " ".join(c["char"] if arabic_ok else c["label"] for c in chars)
    names = " ".join(c["label"] for c in chars)
    strip = size * 2 + 30
    banner = size * 3 + 40
    canvas = Image.new("RGB", (view.width, view.height + strip + banner), (255, 255, 255))
    canvas.paste(view, (0, 0))
    draw = ImageDraw.Draw(canvas)
    line = max(3, view.width // 300)
    draw.rectangle(
        (*to_view(left, top), *to_view(right, bottom)), outline=(0, 0, 0), width=line + 1
    )
    for index, char in enumerate(chars):
        cx1, cy1, cx2, cy2 = char["box"]
        color = _PALETTE[index % len(_PALETTE)]
        bx1, by1 = to_view(left + cx1, top + cy1)
        bx2, by2 = to_view(left + cx2, top + cy2)
        draw.rectangle((bx1, by1, bx2, by2), outline=color, width=line)
        label = char["char"] if arabic_ok else char["label"]
        centre = (bx1 + bx2) / 2
        draw.line((centre, by2, centre, view.height + 4), fill=color, width=max(1, line // 2))
        draw.text((centre, view.height + 8), label, font=font, fill=color, anchor="mt")
        draw.text(
            (centre, view.height + 8 + size + 6),
            f"{char['confidence']:.2f}",
            font=_font(max(16, size // 2))[0],
            fill=color,
            anchor="mt",
        )
    draw.text(
        (view.width / 2, view.height + strip + 10),
        text or "(no characters read)",
        font=font,
        fill=(0, 0, 0),
        anchor="mt",
    )
    draw.text(
        (view.width / 2, view.height + strip + size + 30),
        "classes: " + names,
        font=_font(max(16, size // 2))[0],
        fill=(60, 60, 60),
        anchor="mt",
    )
    return canvas
