import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1] / "scripts"))
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class RecipeTests(unittest.TestCase):
    def test_import_has_no_output(self):
        with tempfile.TemporaryDirectory() as td:
            code = Path(__file__).resolve().parents[1] / "scripts" / "synthetic_plate_cpu.py"
            result = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    f'import runpy; runpy.run_path({str(code)!r}, run_name="library")',
                ],
                cwd=td,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout, "")
            self.assertEqual(list(Path(td).iterdir()), [])

    def test_train_requires_output(self):
        r = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve().parents[1] / "scripts" / "synthetic_plate_cpu.py"),
                "train",
                "--data",
                "missing",
            ],
            capture_output=True,
            text=True,
        )
        self.assertEqual(r.returncode, 2)
        self.assertIn("--output is required", r.stderr)

    def test_alphabet(self):
        import synthetic_plate_cpu as m

        self.assertEqual(len(set(m.LETTERS)), 17)
        self.assertNotIn("ظ", m.LETTERS)
        self.assertNotIn("ك", m.LETTERS)
        self.assertIn("ط", m.LETTERS)
        self.assertIn("ي", m.LETTERS)
        self.assertEqual(len(m.DIGITS), 9)

    def test_missing_font_no_output(self):
        import synthetic_plate_cpu as m

        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "new"
            with self.assertRaisesRegex(ValueError, "Font missing"):
                m.generate(out, ["/missing/font.ttf"])
            self.assertFalse(out.exists())

    def test_invalid_manifest(self):
        import synthetic_plate_cpu as m

        with tempfile.TemporaryDirectory() as td:
            (Path(td) / "manifest.json").write_text("[]")
            with self.assertRaisesRegex(ValueError, "500"):
                m.validate_data(Path(td))

    def test_unsafe_paths(self):
        import synthetic_plate_cpu as m
        import json

        with tempfile.TemporaryDirectory() as td:
            (Path(td) / "manifest.json").write_text(json.dumps([{"file": "../x.png"}] * 500))
            with self.assertRaisesRegex(ValueError, "Unsafe"):
                m.validate_data(Path(td))


if __name__ == "__main__":
    unittest.main()
