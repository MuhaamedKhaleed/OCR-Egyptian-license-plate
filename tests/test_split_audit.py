"""Fictional image fixtures for read-only leakage-screening behavior."""

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "audit_image_splits.py"
spec = importlib.util.spec_from_file_location("split_audit", SCRIPT)
assert spec and spec.loader
audit_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit_module)


class AuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.splits = {name: self.root / name for name in ["train", "dev"]}
        for path in self.splits.values():
            path.mkdir()
            (path / "fictional.jpg").write_bytes(b"fixture, not an image")

    def test_byte_duplicate_without_pillow(self):
        report = audit_module.audit(self.splits)
        self.assertEqual(report["counts"]["same_bytes"], 1)
        self.assertFalse(report["visual"])
        self.assertTrue(report["complete_read"])

    def test_byte_only_does_not_claim_pixel_read(self):
        report = audit_module.audit(self.splits)
        self.assertIsNone(report["dhash_bits"])
        self.assertIsNone(report["findings"][0]["dhash_distance"])

    def test_only_explicit_roots(self):
        (self.root / "test").mkdir()
        (self.root / "test" / "do-not-read.jpg").write_bytes(b"reserved")
        self.assertEqual(audit_module.audit(self.splits)["images"], {"train": 1, "dev": 1})

    def test_limit_before_pair_processing(self):
        (self.splits["dev"] / "second.jpg").write_bytes(b"other")
        with self.assertRaisesRegex(ValueError, "limit"):
            audit_module.audit(self.splits, max_comparisons=1)

    def test_findings_count_not_lost_to_cap(self):
        (self.splits["dev"] / "second.jpg").write_bytes(b"fixture, not an image")
        report = audit_module.audit(self.splits, max_findings=1)
        self.assertEqual(report["counts"]["same_bytes"], 2)
        self.assertEqual(len(report["findings"]), 1)
        self.assertTrue(report["findings_truncated"])

    def test_nested_or_equal_roots_rejected(self):
        for second in [self.splits["train"], self.root]:
            with self.assertRaises(ValueError):
                audit_module.audit({"a": self.splits["train"], "b": second})

    def test_empty_split_rejected(self):
        (self.splits["dev"] / "fictional.jpg").unlink()
        with self.assertRaises(ValueError):
            audit_module.audit(self.splits)

    def test_no_overwrite_cli(self):
        out = self.root / "preserved.json"
        out.write_text("preserved")
        result = self.cli(out)
        self.assertEqual(result.returncode, 2)
        self.assertEqual(out.read_text(), "preserved")

    def cli(self, out, *extra):
        return subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--split",
                f"train={self.splits['train']}",
                "--split",
                f"dev={self.splits['dev']}",
                "--output",
                str(out),
                *extra,
            ],
            capture_output=True,
            text=True,
        )

    def test_cli_roundtrip_and_inputs_unchanged(self):
        before = {str(p): p.read_bytes() for root in self.splits.values() for p in root.iterdir()}
        out = self.root / "audit.json"
        result = self.cli(out)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(out.read_text())["cross_split_comparisons"], 1)
        self.assertEqual(
            before,
            {str(p): p.read_bytes() for root in self.splits.values() for p in root.iterdir()},
        )

    def test_duplicate_split_name_rejected(self):
        result = self.cli(self.root / "out.json", "--split", f"train={self.splits['dev']}")
        self.assertEqual(result.returncode, 2)

    def test_symlink_does_not_read_other_split(self):
        (self.splits["dev"] / "linked.jpg").symlink_to(self.splits["train"] / "fictional.jpg")
        report = audit_module.audit(self.splits)
        self.assertFalse(report["complete_read"])
        self.assertEqual(report["errors"][0]["error"], "unsafe_path")

    def test_help_needs_no_images(self):
        result = subprocess.run([sys.executable, str(SCRIPT), "--help"], capture_output=True)
        self.assertEqual(result.returncode, 0)


try:
    from PIL import Image
except ImportError:
    Image = None


@unittest.skipIf(Image is None, "Optional Pillow visual screening not installed")
class VisualTests(unittest.TestCase):
    setUp = AuditTests.setUp

    def test_same_pixels_different_encoding(self):
        assert Image is not None
        for root in self.splits.values():
            (root / "fictional.jpg").unlink()
        im = Image.new("RGB", (32, 32), (100, 50, 20))
        im.save(self.splits["train"] / "a.png", compress_level=0)
        im.save(self.splits["dev"] / "b.png", compress_level=9)
        report = audit_module.audit(self.splits, visual=True)
        self.assertEqual(report["counts"]["same_pixels"], 1)
        self.assertEqual(report["counts"]["same_bytes"], 0)

    def test_corrupt_image_recorded(self):
        assert Image is not None
        Image.new("RGB", (32, 32)).save(self.splits["train"] / "good.png")
        Image.new("RGB", (32, 32)).save(self.splits["dev"] / "good.png")
        report = audit_module.audit(self.splits, visual=True)
        self.assertFalse(report["complete_read"])
        self.assertEqual(len(report["errors"]), 2)

    def test_perceptual_candidate_not_exact(self):
        assert Image is not None
        for root in self.splits.values():
            (root / "fictional.jpg").unlink()
        Image.new("RGB", (32, 32), (100, 50, 20)).save(self.splits["train"] / "a.png")
        Image.new("RGB", (32, 32), (110, 60, 30)).save(self.splits["dev"] / "b.png")
        report = audit_module.audit(self.splits, visual=True, threshold=0)
        self.assertEqual(report["counts"]["perceptual_candidate"], 1)
        self.assertEqual(report["counts"]["same_pixels"], 0)
