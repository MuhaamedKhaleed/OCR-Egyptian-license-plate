# Egyptian license plate OCR

A YOLO training notebook with a small local-inference package for decoding Egyptian plate character detections. This pilot separates reusable decoding from notebook experiments and fixes reuse of a previous image's prediction.

**Status:** tested package and CPU inference using the original saved checkpoint, not a verified production parking system. Weights are not included. Reviewed Arabic aliases are corrected, but recognition errors remain; retraining, hardware deployment, Firebase and a mobile app have not been verified here.

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

## Local inference

CPU loading and image prediction were verified with the original saved checkpoint. This does not establish OCR accuracy. Install the optional inference dependency, obtain **trusted, compatible local trained weights**, and provide one plate crop:

```sh
python -m pip install -e '.[inference]'
egypt-plate-ocr --image path/to/plate.png --weights path/to/best.pt --confidence 0.2
```

Both paths must exist. Weights are loaded only when inference is requested. Do not load weights from an untrusted source. No automatic weights download is performed by this wrapper. The installed ML libraries may have their own network behavior, so review their settings before use with private images.

Output is UTF-8 JSON with `letters` and `numbers`. An **offline decoder test**, not a model prediction, yields:

```json
{"letters": "أ ب", "numbers": "٣ ١"}
```

The original right-to-left ordering is preserved. Arabic aliases were reviewed against the original saved training annotations and matching character crops, not inferred from their English spellings. Near-identical character boxes (IoU at least 0.8) and nested same-label boxes are reduced to the strongest detection. Unknown classes are ignored. Single-crop mode does not group plates or establish legal plate validity.

For a full vehicle image, opt in to separate localization and recognition:

```sh
egypt-plate-ocr --image path/to/car.jpg --weights path/to/best.pt --confidence 0.2 --auto-crop
```

Auto-crop mode detects `License Plate` boxes, adds a fixed small margin, and predicts each crop independently. It returns a JSON list with `box` (pixel xyxy), detector `confidence`, `letters` and `numbers` for each plate. No detected plate returns `[]`, with no full-frame fallback. Every detected plate is returned, including possible false positives. Confidence is the plate detector score, not a probability that its text is correct.

## Configuration and portable notebook

`model-ocr.ipynb` is now a local, opt-in workflow using the reviewed package. Its default execution runs only an invented decoder example and skips training/image inference. It has no install/download cells, Roboflow calls, Kaggle paths or GPU assumption. Install `uv sync --locked --extra inference --extra dev` before opening it from the repository root.

For deliberate CPU training, set `OCR_RUN_TRAINING=1`, `OCR_DATA_YAML`, `OCR_BASE_WEIGHTS` and a new `OCR_TRAIN_OUTPUT`. `OCR_EPOCHS` defaults to 1. The adapter rejects remote split URLs, download directives, missing directories, identical train/validation directories and output overwrite. It does not certify annotation correctness, image-level disjointness or rights. YAML split lists/text manifests are not supported by this first local adapter. Initial weights must already exist locally. For optional prediction set both `OCR_IMAGE` and `OCR_WEIGHTS`. Use a network-disabled environment when strict offline enforcement is required.

The unused `roboflow_key` function remains only as a backward-compatible config helper, not as a dependency of this workflow. It is not called by the package or notebook. No new credential is needed or requested.

## Architecture

```text
CLI arguments -> local path validation -> YOLO adapter -> character detections
                                                       -> pure decoder -> JSON
```

- `src/egypt_plate_ocr/core.py`: mapping, finite coordinate validation and ordering.
- `src/egypt_plate_ocr/inference.py`: explicit image/weights loading and per-image prediction.
- `src/egypt_plate_ocr/cli.py`: arguments, JSON output and nonzero errors.
- `tests/`: offline tests with a fake model; no claims about ML accuracy.
- `training.py`: explicit local-weights CPU adapter with input/output guards.
- `model-ocr.ipynb`: local opt-in workflow; the old exploratory visualization sequence is replaced, not claimed reproduced.

## Evaluation and limits

The pre-cleanup notebook stored validation results on 341 images: mAP@0.5 approximately 0.886 and mAP@0.5:0.95 approximately 0.543. These were **not rerun** in this pilot. They are detection metrics, not full-plate exact-match recognition accuracy. No 94% recognition claim is established by this work. See `docs/evaluation.md` for the evidence needed next.

The new owned engineering code has a scoped MIT license, not a license grant over inherited code, data, dependencies or models. Confirm their rights before redistribution or deployment. The original Roboflow project displays CC BY 4.0; model and code obligations must still be reviewed separately. No model redistribution is implied.

## Security

A credential was previously committed and the owner clarified it was copied from a third party. It was neither used nor revoked during this work. The current notebook has no credential call or saved outputs. Git history/main and external Kaggle copies are unchanged; cleanup remains a separately coordinated step. Never put a replacement key in this repository.

Ultralytics 8.3.0 is AGPL-3.0 software. Its vendor also states model-related licensing conditions. The package's scoped MIT notice does not override those conditions or grant rights to recovered weights. No paid license was purchased and no private/commercial deployment is approved. See `docs/provenance.md`.

## Next steps

1. Confirm the author's contribution and how this component fits the graduation parking system.
2. Review model/code redistribution rights; recovered compatible weights remain outside this repository.
3. Extend the annotation-grounded mapping audit, investigate missing/confused characters, and evaluate a genuinely held-out plate-level set.
4. Validate real dataset annotations, independent splits and the actual hardware/deployment stack. Local path portability is now implemented but original full-scale results are not reproduced.

No hosted demo or release tag is claimed. CI uses a standard Ubuntu runner only while this repository is public, with no uploaded artifacts or image publishing. Download caching is keyed by uv.lock. If the repository becomes private, its job is skipped. See `docs/verification.md` for results and limits.

## Demo

Real CPU predictions have been checked, but no private image or model has been published. The JSON example above is decoder output, not a real image prediction. See `docs/evaluation.md` for the bounded external test and its failures.

## Verification status

See `docs/verification.md` for exact commands, phase status, assumptions and remaining work. Local Docker is unavailable. The CI workflow builds a multi-stage non-root CLI image and checks help, UID and invalid-input exit status. It intentionally excludes model weights and the optional ML runtime, so it does not validate real OCR inference.

## Container

The CLI smoke image contains runtime dependencies only, not trained weights or Ultralytics. These commands are run in CI:

```sh
docker build --tag egypt-plate-ocr:ci .
docker run --rm egypt-plate-ocr:ci --help
```

No compose file is needed because the CLI has no companion service. Do not use this smoke image as a claim of a working model deployment.


## License scope

New project-owned package/engineering files use the owner's MIT default. See LICENSE and LICENSE-SCOPE.md. This is not permission to relicense historical notebooks, third-party code, datasets, models or weights. Their provenance and terms still need review. Earlier verification entries saying no license was added describe the initial pilot, not this follow-up.

## Checked synthetic wiring demo

See [commands, saved output and scope](docs/synthetic-demo.md). This is invented-input execution evidence, not model accuracy or a real-data gallery.
