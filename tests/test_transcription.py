"""Fictional coordinate fixtures, not image-recognition quality tests."""

import importlib.util
from pathlib import Path
import random
import unittest

from egypt_plate_ocr.core import Detection, decode_characters
from egypt_plate_ocr.transcription import transcription_from_characters


class TranscriptionTests(unittest.TestCase):
    def fixture(self):
        return [
            Detection("baa", 80),
            Detection("alef", 100),
            Detection("seen", 60),
            Detection("1", 10),
            Detection("2", 30),
            Detection("3", 50),
        ]

    def test_opposite_group_orders(self):
        self.assertEqual(
            transcription_from_characters(self.fixture()),
            {"letters_rtl": "أبس", "digits_ltr": "١٢٣"},
        )

    def test_input_permutations_do_not_change_order(self):
        rng = random.Random(20261125)
        detections = self.fixture()
        for _ in range(100):
            rng.shuffle(detections)
            self.assertEqual(
                transcription_from_characters(detections),
                {"letters_rtl": "أبس", "digits_ltr": "١٢٣"},
            )

    def test_legacy_api_is_unchanged(self):
        self.assertEqual(
            decode_characters(self.fixture()), {"letters": "أ ب س", "numbers": "٣ ٢ ١"}
        )

    def test_empty_failure(self):
        self.assertEqual(transcription_from_characters([]), {"letters_rtl": "", "digits_ltr": ""})

    def test_partial_group_preserved(self):
        self.assertEqual(
            transcription_from_characters([Detection("2", -1)]),
            {"letters_rtl": "", "digits_ltr": "٢"},
        )

    def test_aliases_do_not_normalize_other_letters(self):
        result = transcription_from_characters([Detection("alf", 2), Detection("yaa", 1)])
        self.assertEqual(result["letters_rtl"], "أي")

    def test_unknown_and_plate_classes_fail(self):
        for label in ("unknown", "License Plate", "", "ب"):
            with self.assertRaises(ValueError):
                transcription_from_characters([Detection(label, 1)])

    def test_tied_centers_fail_within_group(self):
        for labels in (("baa", "alef"), ("1", "2"), ("baa", "baa")):
            with self.assertRaises(ValueError):
                transcription_from_characters([Detection(label, 1) for label in labels])

    def test_equal_centers_across_groups_are_not_reordered(self):
        self.assertEqual(
            transcription_from_characters([Detection("baa", 1), Detection("1", 1)]),
            {"letters_rtl": "ب", "digits_ltr": "١"},
        )

    def test_oversized_group_fails(self):
        with self.assertRaises(ValueError):
            transcription_from_characters([Detection("baa", i) for i in range(33)])

    def test_evaluator_pairing_contract(self):
        path = Path(__file__).resolve().parents[1] / "scripts" / "plate_text_metrics.py"
        spec = importlib.util.spec_from_file_location("plate_metrics_contract", path)
        assert spec is not None and spec.loader is not None
        metrics = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(metrics)
        expected = {"id": "fictional-1", "letters_rtl": "أبس", "digits_ltr": "١٢٣"}
        output = {"id": "fictional-1", **transcription_from_characters(self.fixture())}
        report = metrics.evaluate({"fictional-1": expected}, {"fictional-1": output})
        self.assertEqual(report["exact_matches"], 1)
        legacy = decode_characters(self.fixture())
        wrong = {
            "id": "fictional-1",
            "letters_rtl": legacy["letters"].replace(" ", ""),
            "digits_ltr": legacy["numbers"].replace(" ", ""),
        }
        self.assertEqual(
            metrics.evaluate({"fictional-1": expected}, {"fictional-1": wrong})[
                "digit_edit_distance"
            ],
            2,
        )


if __name__ == "__main__":
    unittest.main()
