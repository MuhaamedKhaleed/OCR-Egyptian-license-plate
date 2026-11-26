"""Configuration and CLI failure checks with no external network calls."""

from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from egypt_plate_ocr.cli import app
from egypt_plate_ocr.config import roboflow_key


def test_missing_key():
    with patch.dict("os.environ", {}, clear=True):
        with pytest.raises(ValueError, match="Set ROBOFLOW_API_KEY"):
            roboflow_key()


def test_explicit_env_file(tmp_path):
    env = tmp_path / "download.env"
    env.write_text("ROBOFLOW_API_KEY=placeholder-only\n")
    with patch.dict("os.environ", {}, clear=True):
        assert roboflow_key(env) == "placeholder-only"


def test_environment_wins_over_file(tmp_path):
    env = tmp_path / "download.env"
    env.write_text("ROBOFLOW_API_KEY=file-placeholder\n")
    with patch.dict("os.environ", {"ROBOFLOW_API_KEY": "env-placeholder"}, clear=True):
        assert roboflow_key(env) == "env-placeholder"


def test_external_failure_is_clean():
    with patch("egypt_plate_ocr.cli.predict_plate", side_effect=RuntimeError("Model failed")):
        result = CliRunner().invoke(app, ["--image", "a.png", "--weights", "b.pt"])
    assert result.exit_code == 2
    assert "Model failed" in result.output
    assert "Traceback" not in result.output


def test_cli_required_options():
    result = CliRunner().invoke(app, [])
    assert result.exit_code == 2
    assert "Missing option" in result.output
