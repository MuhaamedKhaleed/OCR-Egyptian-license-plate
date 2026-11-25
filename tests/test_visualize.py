"""Rendering test on a synthetic image and hand-built result (no model)."""

import pytest

from egypt_plate_ocr.visualize import annotate_image

Image = pytest.importorskip("PIL.Image", reason="Pillow ships with the inference extra")

PLATES = [
    {
        "box": [20, 20, 180, 80],
        "confidence": 0.9,
        "characters": [
            {"char": "ب", "label": "baa", "confidence": 0.8, "box": [10, 10, 40, 50]},
            {"char": "١", "label": "1", "confidence": 0.7, "box": [60, 10, 90, 50]},
        ],
    }
]


def test_draws_plate_and_character_boxes(tmp_path):
    source, out = tmp_path / "in.png", tmp_path / "out" / "a.png"
    Image.new("RGB", (200, 120), "white").save(source)
    annotate_image(source, PLATES, out)
    picture = Image.open(out).convert("RGB")
    assert picture.getpixel((20, 40)) == (0, 0, 0)  # plate box edge, black
    assert picture.getpixel((30, 30)) != (255, 255, 255)  # first character box edge


def test_rejects_bad_output_name(tmp_path):
    source = tmp_path / "in.png"
    Image.new("RGB", (50, 50)).save(source)
    with pytest.raises(ValueError):
        annotate_image(source, PLATES, tmp_path / "x.txt")
