"""Pure decoding logic, independent of the ML runtime."""

from dataclasses import dataclass
from math import isfinite

ARABIC_MAPPING = {
    "aain": "أ",
    "alef": "ا",
    "alf": "ألف",
    "baa": "ب",
    "daal": "د",
    "dal": "ذ",
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
    "taa": "ت",
    "waaw": "و",
    "waw": "ؤ",
    "yaa": "ي",
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
    """Apply the original notebook mapping and right-to-left ordering.

    Expects characters from ONE plate crop. This is not multi-plate grouping.
    Mapping spellings are preserved, not independently linguistically validated.
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
