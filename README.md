# Egyptian license plate OCR

A YOLO training notebook with a small local-inference package for decoding Egyptian plate character detections. This pilot separates reusable decoding from notebook experiments and fixes reuse of a previous image's prediction.

**Status:** offline unit-tested inference plumbing, not a verified production parking system. Compatible trained weights are not included. Real model inference, retraining, hardware deployment, Firebase and a mobile app have not been verified here.

## Quick start

Use Python 3.11 or later. From the repository root:

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[dev]'
python -m egypt_plate_ocr --help
python -m pytest --cov=egypt_plate_ocr --cov-report=term-missing
python -m ruff check src tests
python -m ruff format --check src tests
python -m mypy src
python -m pre_commit run --all-files
```

The package, help, and offline tests need no API keys, model download, paid service or GPU. Initial dependency installation requires internet access. Development dependency versions are in `pyproject.toml`; `uv.lock` records resolution. With uv installed, `uv sync --locked --extra dev` is an alternative.

## Real inference (not yet validated with project weights)

The following is an intended usage path, not a claim of successful model execution. Install the optional inference dependency, obtain **trusted, compatible local trained weights**, and provide one plate crop:

```sh
python -m pip install -e '.[inference]'
egypt-plate-ocr --image path/to/plate.png --weights path/to/best.pt --confidence 0.2
```

Both paths must exist. Weights are loaded only when inference is requested. Do not load weights from an untrusted source. No automatic weights download is performed by this wrapper. The installed ML libraries may have their own network behavior, so review their settings before use with private images.

Output is UTF-8 JSON with `letters` and `numbers`. An **offline decoder test**, not a model prediction, yields:

```json
{"letters": "ا ب", "numbers": "٣ ١"}
```

The original class mapping and right-to-left ordering are preserved. Unknown classes are ignored. This decoder assumes a single plate crop; it does not group multiple plates or establish legal plate validity. Mapping aliases and digit order still require review against the actual dataset.

## Configuration

Local inference requires no key. The optional `roboflow_key(Path(".env"))` helper loads an explicit dotenv file without overriding environment values and raises a clear error when `ROBOFLOW_API_KEY` is missing. It is not called during local inference. The original notebook still uses the environment directly.

## Architecture

```text
CLI arguments -> local path validation -> YOLO adapter -> character detections
                                                       -> pure decoder -> JSON
```

- `src/egypt_plate_ocr/core.py`: mapping, finite coordinate validation and ordering.
- `src/egypt_plate_ocr/inference.py`: explicit image/weights loading and per-image prediction.
- `src/egypt_plate_ocr/cli.py`: arguments, JSON output and nonzero errors.
- `tests/`: offline tests with a fake model; no claims about ML accuracy.
- `model-ocr.ipynb`: original experimental training workflow with outputs cleared, key externalized and final prediction corrected. Kaggle paths remain; it is **not** a portable training pipeline yet.

## Evaluation and limits

The pre-cleanup notebook stored validation results on 341 images: mAP@0.5 approximately 0.886 and mAP@0.5:0.95 approximately 0.543. These were **not rerun** in this pilot. They are detection metrics, not full-plate exact-match recognition accuracy. No 94% recognition claim is established by this work. See `docs/evaluation.md` for the evidence needed next.

No default license has been added. Confirm rights for the original code, dataset and trained weights before redistribution or deployment. The dataset's provenance is referenced in the notebook, but its license and redistribution terms have not been verified here.

## Security

A credential was previously committed. The current notebook reads `ROBOFLOW_API_KEY` from the environment and contains no saved outputs. This **does not revoke the old key or erase git history**. Revoke the exposed key at the provider and handle history cleanup separately with coordination for any collaborators. Do not put a replacement key in a notebook or commit `.env`.

## Next steps

1. Confirm the author's contribution and how this component fits the graduation parking system.
2. Verify dataset/model rights and provide compatible weights through a safe route.
3. Run real inference, record expected outputs and failure examples, and test a held-out plate-level set.
4. Make training paths configurable and validate the actual hardware/deployment stack before describing this as production-ready.

No Docker image, hosted demo, release tag or CI badge is claimed. No hosted CI workflow is added in this no-fee pilot; run checks locally to avoid triggering billable jobs.

## Demo

A real-model screenshot or recording is not available yet. Add one only after validating compatible weights on representative plate images. The JSON example above is decoder output, not a real image prediction.

## Verification status

See `docs/verification.md` for exact commands, phase status, assumptions and remaining work. Docker is not installed in the verification environment, so no image build/run is claimed.
