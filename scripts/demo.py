"""Explicit synthetic wiring demo. No real-data accuracy or quality claim."""

import json
import os
from pathlib import Path
import sys


def main() -> None:
    for key, value in {
        "CUDA_VISIBLE_DEVICES": "-1",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "TF_CPP_MIN_LOG_LEVEL": "3",
        "TF_NUM_INTRAOP_THREADS": "1",
        "TF_NUM_INTEROP_THREADS": "1",
        "OMP_NUM_THREADS": "1",
    }.items():
        os.environ.setdefault(key, value)

    out = Path(sys.argv[1] if len(sys.argv) == 2 else "synthetic-run")
    if len(sys.argv) > 2:
        raise SystemExit("Usage: python scripts/demo.py [NEW_OUTPUT_DIRECTORY]")
    out.mkdir(parents=True, exist_ok=False)
    from egypt_plate_ocr.core import Detection, decode_characters

    result = decode_characters(
        [Detection("baa", 10), Detection("alef", 30), Detection("3", 40), Detection("1", 20)]
    )
    assert result == {"letters": "أ ب", "numbers": "٣ ١"}
    scope = "invented_character_boxes_decoder_only_not_detector_or_real_OCR"
    report = {"synthetic_only": True, "scope": scope, "quality_claim": False, "result": result}
    (out / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(
        json.dumps({"synthetic_only": True, "report": str(out / "report.json")}, ensure_ascii=False)
    )


if __name__ == "__main__":
    main()
