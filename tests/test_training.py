"""Local training guards and notebook default execution."""

import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from egypt_plate_ocr.training import train_local


def test_local_training_cpu_and_no_download(tmp_path):
    data, weights = tmp_path / "data.yaml", tmp_path / "base.pt"
    data.write_text("train: images/train\nval: images/val\n")
    (tmp_path / "images/train").mkdir(parents=True)
    (tmp_path / "images/val").mkdir()
    weights.write_bytes(b"fake fixture, never deserialized")
    factory = Mock()
    result = train_local(data, weights, tmp_path / "run", model_factory=factory)
    factory.assert_called_once_with(str(weights.resolve()))
    options = factory.return_value.train.call_args.kwargs
    assert options["device"] == "cpu"
    assert options["workers"] == 0
    assert options["pretrained"] is False
    assert options["exist_ok"] is False
    assert result["device"] == "cpu"


@pytest.mark.parametrize("case", ["missing_data", "missing_weights", "existing", "epochs", "size"])
def test_training_guards(tmp_path, case):
    data, weights = tmp_path / "data.yaml", tmp_path / "base.pt"
    data.write_text("train: images/train\nval: images/val\n")
    weights.touch()
    output = tmp_path / "run"
    if case == "missing_data":
        data.unlink()
    elif case == "missing_weights":
        weights.unlink()
    elif case == "existing":
        output.mkdir()
    factory = Mock()
    with pytest.raises(ValueError):
        train_local(
            data,
            weights,
            output,
            epochs=0 if case == "epochs" else 1,
            image_size=33 if case == "size" else 64,
            model_factory=factory,
        )
    factory.assert_not_called()


def test_notebook_safe_default(monkeypatch):
    for key in ("OCR_RUN_TRAINING", "OCR_IMAGE", "OCR_WEIGHTS"):
        monkeypatch.delenv(key, raising=False)
    notebook = json.loads(Path("model-ocr.ipynb").read_text())
    namespace = {"__name__": "__main__"}
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            assert not cell.get("outputs")
            exec(compile("".join(cell["source"]), "model-ocr.ipynb", "exec"), namespace)
    assert namespace["decoder_result"] == {"letters": "ب", "numbers": "١"}


@pytest.mark.parametrize(
    "content",
    [
        "download: echo unsafe\ntrain: train\nval: val\n",
        "train: https://example.invalid/train\nval: val\n",
        "train: missing\nval: missing\n",
        "- invalid-mapping\n",
    ],
)
def test_dataset_download_and_invalid_paths_refused(tmp_path, content):
    data, weights = tmp_path / "data.yaml", tmp_path / "base.pt"
    data.write_text(content)
    weights.touch()
    factory = Mock()
    with pytest.raises(ValueError):
        train_local(data, weights, tmp_path / "run", model_factory=factory)
    factory.assert_not_called()
