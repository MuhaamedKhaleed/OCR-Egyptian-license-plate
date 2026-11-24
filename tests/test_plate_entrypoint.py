import importlib.util
from pathlib import Path
import subprocess
import sys


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/plate_localization_cpu.py"


def test_safe_import():
    spec = importlib.util.spec_from_file_location("plate_experiment", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert callable(module.main)


def test_help_without_training():
    r = subprocess.run([sys.executable, str(SCRIPT), "--help"], capture_output=True, text=True)
    assert r.returncode == 0
    assert "--weights" in r.stdout


def test_requires_local_weights(tmp_path):
    r = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--data",
            str(tmp_path),
            "--weights",
            str(tmp_path / "absent.pt"),
            "--output",
            str(tmp_path / "new"),
        ],
        capture_output=True,
        text=True,
    )
    assert r.returncode != 0
    assert not (tmp_path / "new").exists()
