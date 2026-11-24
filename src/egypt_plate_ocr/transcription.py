"""Explicit coordinate-to-text adapter for the strict evaluation convention."""

from .core import ARABIC_MAPPING, Detection


def transcription_from_characters(detections: list[Detection]) -> dict[str, str]:
    """Return unspaced logical RTL letters and visual LTR digits for one crop.

    This consumes character detections, not legacy decoded strings. Unknown
    classes and tied centers within a group fail rather than drop or reorder
    ambiguous detections. The caller must localize one single-row plate and
    remove duplicate boxes first. Empty input is an explicit recognition failure.
    No confidence calibration, spelling correction or plate-rule repair occurs.
    """
    letters: list[tuple[float, str]] = []
    digits: list[tuple[float, str]] = []
    for detection in detections:
        if detection.label not in ARABIC_MAPPING:
            raise ValueError("Expected character classes only; unknown class detected.")
        character = ARABIC_MAPPING[detection.label]
        group = digits if character.isdecimal() else letters
        group.append((detection.center_x, character))
    for group in (letters, digits):
        if len({center for center, _ in group}) != len(group):
            raise ValueError("Tied character centers make group order ambiguous.")
        if len(group) > 32:
            raise ValueError("Character group exceeds the evaluator's 32-character limit.")
    return {
        "letters_rtl": "".join(char for _, char in sorted(letters, reverse=True)),
        "digits_ltr": "".join(char for _, char in sorted(digits)),
    }
