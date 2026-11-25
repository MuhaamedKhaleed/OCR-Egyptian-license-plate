"""Public-detector option tests. No real images or weights."""

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


def plate_model(calls, names):
    class Model:
        def __init__(self, weights):
            pass

        def predict(self, **kwargs):
            calls.append(kwargs["source"])
            if len(calls) == 1:
                return [
                    SimpleNamespace(orig_img=Image(), names=names, boxes=[box(0, [10, 20, 70, 50])])
                ]
            return [SimpleNamespace(names={0: "baa"}, boxes=[box(0, [5, 5, 15, 25])])]

    return Model


class PublicDetectorTests(TestCase):
    def files(self, folder):
        image, weights = Path(folder) / "i", Path(folder) / "w"
        image.touch()
        weights.touch()
        return image, weights

    def test_custom_detector_label_localizes(self):
        calls = []
        with TemporaryDirectory() as folder:
            image, weights = self.files(folder)
            result = predict_plates(
                image,
                weights,
                model_factory=plate_model(calls, {0: "license_plate"}),
                detector_label="license_plate",
            )
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["box"], [7, 17, 74, 54])
        self.assertEqual(result[0]["letters"], "ب")
        self.assertEqual(len(calls), 2)

    def test_default_label_skips_public_detector_class(self):
        calls = []
        with TemporaryDirectory() as folder:
            image, weights = self.files(folder)
            result = predict_plates(
                image, weights, model_factory=plate_model(calls, {0: "license_plate"})
            )
        self.assertEqual(result, [])
        self.assertEqual(len(calls), 1)

    def test_localize_only_returns_boxes_without_decoding(self):
        calls = []
        with TemporaryDirectory() as folder:
            image, weights = self.files(folder)
            result = predict_plates(
                image,
                weights,
                model_factory=plate_model(calls, {0: "license_plate"}),
                detector_label="license_plate",
                localize_only=True,
            )
        self.assertEqual(result, [{"box": [7, 17, 74, 54], "confidence": 0.9}])
        self.assertEqual(len(calls), 1)
