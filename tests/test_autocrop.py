"""Synthetic multi-plate tests. No real images or weights."""

from pathlib import Path
from types import SimpleNamespace
from tempfile import TemporaryDirectory
from unittest import TestCase

from egypt_plate_ocr.inference import predict_plates


class Scalar:
    def __init__(self, value):
        self.value = value

    def item(self):
        return self.value


class Coords:
    def __init__(self, value):
        self.value = value

    def tolist(self):
        return self.value


def box(cls, xyxy, confidence=0.9):
    return SimpleNamespace(cls=Scalar(cls), conf=Scalar(confidence), xyxy=[Coords(xyxy)])


class Image:
    shape = (100, 200, 3)

    def __getitem__(self, key):
        return ("crop", key)


class AutoCropTests(TestCase):
    def test_separate_plates_never_merge_characters(self):
        calls = []

        class Model:
            def __init__(self, weights):
                pass

            def predict(self, **kwargs):
                calls.append(kwargs["source"])
                if len(calls) == 1:
                    return [
                        SimpleNamespace(
                            orig_img=Image(),
                            names={0: "License Plate"},
                            boxes=[box(0, [10, 20, 70, 50]), box(0, [100, 20, 150, 50])],
                        )
                    ]
                return [
                    SimpleNamespace(
                        names={0: "baa" if len(calls) == 2 else "1"}, boxes=[box(0, [5, 5, 15, 25])]
                    )
                ]

        with TemporaryDirectory() as folder:
            image, weights = Path(folder) / "i", Path(folder) / "w"
            image.touch()
            weights.touch()
            result = predict_plates(image, weights, model_factory=Model)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["letters"], "ب")
        self.assertEqual(result[0]["numbers"], "")
        self.assertEqual(result[1]["letters"], "")
        self.assertEqual(result[1]["numbers"], "١")
        self.assertEqual(result[0]["box"], [7, 17, 74, 54])

    def test_no_plate_returns_empty_not_full_frame(self):
        class Model:
            def __init__(self, weights):
                pass

            def predict(self, **kwargs):
                return [
                    SimpleNamespace(
                        orig_img=Image(), names={0: "car"}, boxes=[box(0, [1, 2, 3, 4])]
                    )
                ]

        with TemporaryDirectory() as folder:
            p = Path(folder) / "file"
            p.touch()
            self.assertEqual(predict_plates(p, p, model_factory=Model), [])

    def test_invalid_inputs(self):
        for confidence in [-1, 2, float("nan")]:
            with self.assertRaises(ValueError):
                predict_plates(Path("none"), Path("none"), confidence)
        with self.assertRaises(ValueError):
            predict_plates(Path("none"), Path("none"))


class CharacterBoxTests(TestCase):
    def _run(self, **kwargs):
        calls = []
        self.kws = kws = []

        class Model:
            def __init__(self, weights):
                pass

            def predict(self, **kw):
                calls.append(kw["source"])
                kws.append(kw)
                if len(calls) == 1:
                    return [
                        SimpleNamespace(
                            orig_img=Image(),
                            names={0: "License Plate"},
                            boxes=[box(0, [10, 20, 70, 50])],
                        )
                    ]
                return [
                    SimpleNamespace(
                        names={0: "baa", 1: "1"},
                        boxes=[box(0, [5, 5, 15, 25], 0.8), box(1, [30, 5, 40, 25], 0.7)],
                    )
                ]

        with TemporaryDirectory() as tmp:
            image, weights = Path(tmp, "i.jpg"), Path(tmp, "w.pt")
            image.write_bytes(b"x")
            weights.write_bytes(b"x")
            return predict_plates(image, weights, model_factory=Model, **kwargs)

    def test_default_output_has_no_boxes(self):
        (plate,) = self._run()
        self.assertNotIn("characters", plate)
        self.assertEqual(plate["numbers"], "١")
        self.assertEqual(plate["letters"], "ب")

    def test_boxes_are_the_models_own_detections_right_to_left(self):
        (plate,) = self._run(include_boxes=True)
        self.assertEqual([c["char"] for c in plate["characters"]], ["١", "ب"])
        self.assertEqual(plate["characters"][0]["box"], [30, 5, 40, 25])
        self.assertEqual(plate["characters"][1]["label"], "baa")
        self.assertEqual(plate["numbers"], "١")

    def test_boxes_with_localize_only_rejected(self):
        with self.assertRaises(ValueError):
            self._run(include_boxes=True, localize_only=True)

    def test_tta_is_off_by_default_and_only_on_the_character_pass(self):
        self._run()
        self.assertNotIn("augment", self.kws[0])
        self.assertNotIn("augment", self.kws[1])
        self._run(tta=True)
        self.assertNotIn("augment", self.kws[0])
        self.assertIs(self.kws[1]["augment"], True)

    def test_tta_with_localize_only_rejected(self):
        with self.assertRaises(ValueError):
            self._run(tta=True, localize_only=True)


class FrameSourceTests(TestCase):
    """char_source="frame": characters come from the one full-frame pass."""

    def _run(self, **kwargs):
        self.calls = calls = []

        class Model:
            def __init__(self, weights):
                pass

            def predict(self, **kw):
                calls.append(kw["source"])
                return [
                    SimpleNamespace(
                        orig_img=Image(),
                        names={0: "License Plate", 1: "baa", 2: "1", 3: "car"},
                        boxes=[
                            box(0, [10, 20, 70, 50]),
                            box(1, [50, 25, 60, 45], 0.8),
                            box(2, [20, 25, 30, 45], 0.7),
                            box(1, [150, 25, 160, 45], 0.9),  # outside the plate
                            box(3, [0, 0, 200, 100], 0.9),  # not a character class
                        ],
                    )
                ]

        with TemporaryDirectory() as tmp:
            image, weights = Path(tmp, "i.jpg"), Path(tmp, "w.pt")
            image.write_bytes(b"x")
            weights.write_bytes(b"x")
            return predict_plates(
                image, weights, model_factory=Model, char_source="frame", **kwargs
            )

    def test_uses_one_pass_and_only_characters_inside_the_plate(self):
        (plate,) = self._run(include_boxes=True)
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(plate["letters"], "ب")
        self.assertEqual(plate["numbers"], "١")
        self.assertEqual(len(plate["characters"]), 2)

    def test_character_boxes_are_relative_to_the_plate_crop(self):
        (plate,) = self._run(include_boxes=True)
        left, top = plate["box"][0], plate["box"][1]
        by_label = {c["label"]: c["box"] for c in plate["characters"]}
        self.assertEqual(by_label["baa"], [50 - left, 25 - top, 60 - left, 45 - top])

    def test_tta_and_unknown_source_rejected(self):
        with self.assertRaises(ValueError):
            self._run(tta=True)
        with self.assertRaises(ValueError):
            predict_plates(Path("a"), Path("b"), char_source="nope")
