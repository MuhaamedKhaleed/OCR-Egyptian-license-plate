"""Offline regression checks for the decoding contract and input safety."""

import math
import contextlib
import io
from unittest.mock import patch
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from egypt_plate_ocr.cli import main
from egypt_plate_ocr.core import Detection, decode_characters
from egypt_plate_ocr.inference import predict_plate


class CoreTests(unittest.TestCase):
    def test_empty(self):
        self.assertEqual(decode_characters([]), {"letters": "", "numbers": ""})

    def test_mapping_and_rtl(self):
        chars = [
            Detection("baa", 10),
            Detection("alef", 30),
            Detection("3", 40),
            Detection("1", 20),
            Detection("plate", 50),
        ]
        self.assertEqual(decode_characters(chars), {"letters": "ا ب", "numbers": "٣ ١"})

    def test_unknown(self):
        self.assertEqual(decode_characters([Detection("unknown", 1)])["letters"], "")

    def test_nonfinite(self):
        for x in [math.nan, math.inf, -math.inf]:
            with self.assertRaises(ValueError):
                Detection("1", x)

    def test_missing_files(self):
        with self.assertRaises(ValueError):
            predict_plate(Path("/not-an-image"), Path("/not-weights"))

    def test_invalid_confidence(self):
        for x in [-0.1, 1.1, math.nan]:
            with self.assertRaises(ValueError):
                predict_plate(Path("a"), Path("b"), x)

    def test_cli_missing_image(self):
        self.assertEqual(main(["--image", "/not-an-image", "--weights", "/not-weights"]), 2)

    def test_missing_weights(self):
        with TemporaryDirectory() as d:
            image = Path(d) / "image.png"
            image.touch()
            with self.assertRaisesRegex(ValueError, "Weights not found"):
                predict_plate(image, Path(d) / "missing.pt")

    def test_empty_model_results(self):
        class EmptyModel:
            def __init__(self, weights):
                pass

            def predict(self, **kwargs):
                return []

        with TemporaryDirectory() as d:
            image, weights = Path(d) / "image.png", Path(d) / "weights.pt"
            image.touch()
            weights.touch()
            self.assertEqual(
                predict_plate(image, weights, model_factory=EmptyModel),
                {"letters": "", "numbers": ""},
            )

    def test_cli_json_output(self):
        output = io.StringIO()
        with patch(
            "egypt_plate_ocr.cli.predict_plate", return_value={"letters": "ب", "numbers": ""}
        ):
            with contextlib.redirect_stdout(output):
                status = main(["--image", "image.png", "--weights", "weights.pt"])
        self.assertEqual(status, 0)
        self.assertIn("ب", output.getvalue())

    def test_cli_help_exits_zero(self):
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(["--help"]), 0)

    def test_current_image_always_predicts(self):
        calls = []

        class Scalar:
            def item(self):
                return 0

        class Coords:
            def tolist(self):
                return [0, 0, 10, 10]

        class FakeModel:
            def __init__(self, weights):
                self.weights = weights

            def predict(self, **kwargs):
                calls.append(kwargs)
                return [
                    SimpleNamespace(
                        boxes=[SimpleNamespace(cls=Scalar(), xyxy=[Coords()])], names={0: "baa"}
                    )
                ]

        with TemporaryDirectory() as d:
            root = Path(d)
            weights = root / "test.pt"
            weights.touch()
            for name in ["first.png", "second.png"]:
                image = root / name
                image.touch()
                result = predict_plate(image, weights, model_factory=FakeModel)
                self.assertEqual(result["letters"], "ب")
            self.assertTrue(calls[0]["source"].endswith("first.png"))
            self.assertTrue(calls[1]["source"].endswith("second.png"))
            self.assertNotEqual(calls[0]["source"], calls[1]["source"])


if __name__ == "__main__":
    unittest.main()
