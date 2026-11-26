"""Strict paired transcription metrics, not image inference or a dataset audit."""

from __future__ import annotations
import argparse
import json
from pathlib import Path


def edit_distance(a: str, b: str) -> int:
    """Unit-cost insertion/deletion/substitution Levenshtein distance."""
    previous = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        current = [i]
        for j, y in enumerate(b, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (x != y)))
        previous = current
    return previous[-1]


def load_records(path: Path, *, truth: bool) -> dict[str, dict[str, str]]:
    """JSON list of ID, logical RTL letter order, visual LTR digit order.

    Strings must already use the intended exact characters. No spelling,
    hamza, digit-system, ordering, whitespace or confusable normalization.
    Empty predictions represent failures; empty ground truth is rejected.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list) or not data:
        raise ValueError("Need a nonempty JSON list.")
    result = {}
    for r in data:
        if not isinstance(r, dict) or set(r) != {"id", "letters_rtl", "digits_ltr"}:
            raise ValueError("Each row needs exactly id, letters_rtl and digits_ltr.")
        if any(not isinstance(r[k], str) for k in r):
            raise ValueError("Every field must be a string.")
        if not r["id"] or r["id"] in result:
            raise ValueError("IDs must be nonempty and unique.")
        for key in ["letters_rtl", "digits_ltr"]:
            if any(c.isspace() for c in r[key]):
                raise ValueError("Transcriptions must not contain whitespace.")
        if truth and (not r["letters_rtl"] or not r["digits_ltr"]):
            raise ValueError("Ground truth needs both nonempty letters and digits.")
        if any(not c.isalpha() for c in r["letters_rtl"]):
            raise ValueError("Letters field must contain letters only.")
        if any(not c.isdecimal() for c in r["digits_ltr"]):
            raise ValueError("Digits field must contain decimal digits only.")
        if len(r["letters_rtl"]) > 32 or len(r["digits_ltr"]) > 32:
            raise ValueError("Field length exceeds32characters.")
        result[r["id"]] = r
    return result


def evaluate(truth: dict[str, dict[str, str]], predicted: dict[str, dict[str, str]]) -> dict:
    """Every truth ID must have an explicit prediction, including failures.

    A missed whole plate is an explicit pair of empty prediction strings.
    Missing prediction rows fail instead of inflating success by exclusion.
    """
    if not truth or set(truth) != set(predicted):
        raise ValueError("Truth and prediction IDs must match exactly.")
    rows = []
    letter_edits = digit_edits = letter_chars = digit_chars = exact = 0
    for ident, t in truth.items():
        p = predicted[ident]
        ld = edit_distance(t["letters_rtl"], p["letters_rtl"])
        dd = edit_distance(t["digits_ltr"], p["digits_ltr"])
        le = len(t["letters_rtl"])
        de = len(t["digits_ltr"])
        if not le or not de:
            raise ValueError("Ground truth must not be empty.")
        match = ld == 0 and dd == 0
        exact += match
        letter_edits += ld
        digit_edits += dd
        letter_chars += le
        digit_chars += de
        # No raw registration text copied into result.
        rows.append(
            {
                "id": ident,
                "exact": match,
                "letter_edits": ld,
                "digit_edits": dd,
                "truth_characters": le + de,
                "all_prediction_empty": not p["letters_rtl"] and not p["digits_ltr"],
            }
        )
    return {
        "plates": len(rows),
        "exact_matches": exact,
        "plate_exact_match": exact / len(rows),
        "letter_edit_distance": letter_edits,
        "digit_edit_distance": digit_edits,
        "truth_letters": letter_chars,
        "truth_digits": digit_chars,
        "letter_error_rate": letter_edits / letter_chars,
        "digit_error_rate": digit_edits / digit_chars,
        "groupwise_character_error_rate": (letter_edits + digit_edits)
        / (letter_chars + digit_chars),
        "empty_predictions": sum(r["all_prediction_empty"] for r in rows),
        "rows": rows,
        "scope": [
            "Text pairing only, no image inference or visual label verification.",
            "Logical RTL letters and visual LTR digits supplied explicitly; no automatic order correction.",
            "Exact Unicode text, no normalization. Arabic yeh/alefmaksura/hamza/digit-script differences count.",
            "Groupwise character edit metric, not full mixed-script string CER.",
            "Edit rates can exceed1because insertions count.",
            "No independent-test, vehicle-separation or privacy guarantee follows from metrics.",
        ],
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--truth", type=Path, required=True)
    p.add_argument("--predictions", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    try:
        if a.output.exists():
            raise ValueError("Output exists; choose a new file.")
        r = evaluate(load_records(a.truth, truth=True), load_records(a.predictions, truth=False))
        # Atomic no-overwrite at commit, not just the earlier exists check.
        with a.output.open("x", encoding="utf-8") as f:
            json.dump(r, f, ensure_ascii=False, indent=2, allow_nan=False)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        p.exit(2, f"Error: {exc}\n")
    print(
        json.dumps({k: v for k, v in r.items() if k != "rows"}, ensure_ascii=False, allow_nan=False)
    )


if __name__ == "__main__":
    main()
