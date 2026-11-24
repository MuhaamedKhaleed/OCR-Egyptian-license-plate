"""Pure decoding logic, independent of the ML runtime."""

from dataclasses import dataclass
from math import isfinite

ARABIC_MAPPING = {
    "aain": "ع",
    "alef": "أ",
    "alf": "أ",
    "baa": "ب",
    "daal": "د",
    "dad": "ض",
    "dal": "د",
    "e": "ع",
    "faa": "ف",
    "geem": "ج",
    "haa": "ه",
    "kaaf": "ك",
    "laam": "ل",
    "meem": "م",
    "noon": "ن",
    "qaf": "ق",
    "raa": "ر",
    "sad": "ص",
    "seen": "س",
    "taa": "ط",
    "waaw": "و",
    "waw": "و",
    "yaa": "ي",
    "zay": "ز",
    **dict(zip("0123456789", "٠١٢٣٤٥٦٧٨٩", strict=True)),
}


@dataclass(frozen=True)
class Detection:
    """A detected character class and its horizontal center in image pixels."""

    label: str
    center_x: float

    def __post_init__(self) -> None:
        if not isfinite(self.center_x):
            raise ValueError("Detection center must be finite.")


def decode_characters(detections: list[Detection]) -> dict[str, str]:
    """Apply the audited dataset mapping and right-to-left ordering.

    Expects characters from ONE plate crop. This is not multi-plate grouping.
    Ambiguous aliases were reviewed against original labeled character images.
    Unknown classes (including plate bounding boxes) are ignored.
    """
    chars = [
        ARABIC_MAPPING[d.label]
        for d in sorted(detections, key=lambda d: d.center_x, reverse=True)
        if d.label in ARABIC_MAPPING
    ]
    letters = " ".join(c for c in chars if not c.isdigit())
    numbers = " ".join(c for c in chars if c.isdigit())
    return dict(letters=letters, numbers=numbers)
