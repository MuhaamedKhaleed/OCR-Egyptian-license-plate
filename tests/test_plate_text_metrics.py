import unittest
import tempfile
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from plate_text_metrics import edit_distance, load_records, evaluate


class MetricsTests(unittest.TestCase):
    def row(self, letters="أبس", digits="١٢٣"):
        return {"id": "x", "letters_rtl": letters, "digits_ltr": digits}

    def test_edit_cases(self):
        for a, b, expected in [
            ("", "", 0),
            ("abc", "", 3),
            ("abc", "adc", 1),
            ("abc", "ac", 1),
            ("a", "aaaa", 3),
            ("١", "1", 1),
        ]:
            self.assertEqual(edit_distance(a, b), expected)

    def test_perfect(self):
        a = {"x": self.row()}
        r = evaluate(a, a)
        self.assertEqual(r["plate_exact_match"], 1)
        self.assertEqual(r["groupwise_character_error_rate"], 0)

    def test_empty_prediction_penalized(self):
        r = evaluate({"x": self.row()}, {"x": self.row("", "")})
        self.assertEqual(r["groupwise_character_error_rate"], 1)
        self.assertEqual(r["empty_predictions"], 1)

    def test_insertions_exceed_one(self):
        r = evaluate({"x": self.row("أ", "١")}, {"x": self.row("أ" * 5, "١" * 5)})
        self.assertEqual(r["groupwise_character_error_rate"], 4)

    def test_missing_and_extra_ids_rejected(self):
        for b in [{}, {"y": self.row()}]:
            with self.assertRaises(ValueError):
                evaluate({"x": self.row()}, b)

    def test_order_not_repaired(self):
        r = evaluate({"x": self.row()}, {"x": self.row("سبأ", "٣٢١")})
        self.assertEqual(r["exact_matches"], 0)

    def test_no_raw_text_in_output(self):
        r = evaluate({"x": self.row()}, {"x": self.row()})
        self.assertNotIn("أبس", json.dumps(r, ensure_ascii=False))

    def test_unicode_not_normalized(self):
        r = evaluate({"x": self.row("أ", "١")}, {"x": self.row("ا", "1")})
        self.assertEqual(r["letter_edit_distance"], 1)
        self.assertEqual(r["digit_edit_distance"], 1)

    def load(self, rows, truth=True):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "x.json"
            p.write_text(json.dumps(rows))
            return load_records(p, truth=truth)

    def test_duplicate_rejected(self):
        with self.assertRaises(ValueError):
            self.load([self.row(), self.row()])

    def test_schema_rejected(self):
        for rows in [
            {},
            [],
            [{"id": "x"}],
            [self.row("أ ب")],
            [self.row("", "١")],
            [self.row("ب", "x")],
        ]:
            with self.assertRaises(ValueError):
                self.load(rows)

    def test_prediction_empty_valid(self):
        self.assertEqual(len(self.load([self.row("", "")], False)), 1)

    def test_cli_no_overwrite(self):
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            a = d / "a.json"
            a.write_text(json.dumps([self.row()]))
            out = d / "out.json"
            out.write_text("preserve")
            r = subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).resolve().parents[1] / "scripts" / "plate_text_metrics.py"),
                    "--truth",
                    str(a),
                    "--predictions",
                    str(a),
                    "--output",
                    str(out),
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(r.returncode, 2)
            self.assertEqual(out.read_text(), "preserve")


if __name__ == "__main__":
    unittest.main()
