"""Explicit public detector download. No import-time network or weights redistribution."""

import argparse
import hashlib
import json
import shutil
import urllib.request
from pathlib import Path

# omasayeddd/egyptian-license-plate-detector at the pinned revision. License: MIT
# per the model card front matter and the Hugging Face API license field at fetch
# time (25 November 2026). Single class "license_plate": plate localization only,
# no character recognition. Served bytes verified: the sha256 below equals the
# x-linked-etag Hugging Face returned for best.pt at this exact revision.
# The model card states training on the EALPR Egyptian Vehicles dataset; the
# EALPR GitHub repository ships no LICENSE file (citation request only), so
# dataset-level rights stay the reviewer's responsibility. Ultralytics remains
# AGPL-3.0 software; see README and docs/provenance.md.
REVISION = "b9c1ace323f96988f0ee58106a7942b05536c007"
BASE = f"https://huggingface.co/omasayeddd/egyptian-license-plate-detector/resolve/{REVISION}"

SOURCES = [
    (
        "egyptian-plate-detector.pt",
        f"{BASE}/best.pt",
        "c8f4fbee9040cf316a2bf3649689d55db8a845f83151d85c8b82a1525a5690a6",
    ),
    (
        "detector-config.json",
        f"{BASE}/config.json",
        "a28afa4049c61cd0e3a26f91bfd5f444730ff063df4e0628bce64867c26bcfc6",
    ),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path, help="New detector directory")
    parser.add_argument("--cached", type=Path, help="Use already downloaded files, without network")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = []
    for filename, url, sha in SOURCES:
        target = args.output / filename
        if args.cached:
            shutil.copyfile(args.cached / filename, target)
        else:
            with (
                urllib.request.urlopen(url, timeout=300) as source,
                target.open("wb") as out,
            ):
                shutil.copyfileobj(source, out)
        actual = hashlib.sha256(target.read_bytes()).hexdigest()
        if actual != sha:
            target.unlink()
            raise ValueError("Source checksum mismatch: " + filename)
        manifest.append({"file": filename, "url": url, "sha256": actual})
    (args.output / "detector-manifest.json").write_text(
        json.dumps(
            {
                "model": "omasayeddd/egyptian-license-plate-detector",
                "revision": REVISION,
                "license": "mit",
                "license_source": "model card front matter and HF API license field, 2026-11-25",
                "scope": "plate localization only, single class license_plate",
                "files": manifest,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
