# Egyptian license plate OCR

A YOLO training notebook with a small local-inference package for decoding Egyptian plate character detections. This pilot separates reusable decoding from notebook experiments and fixes reuse of a previous image's prediction.

**Status:** tested package and CPU inference using the original saved checkpoint, not a verified production parking system. Weights are not included. Reviewed Arabic aliases are corrected, but recognition errors remain; retraining, hardware deployment, Firebase and a mobile app have not been verified here.

## Fork-to-output paths (what works without private files)

Needs Python 3.11 or 3.12 (`uv sync --python 3.11 --locked --extra inference --extra dev`; the locked PyTorch wheels have no 3.13+ build) and about 2 GB of RAM.

1. **Plate localization, works today from a fresh clone** (pinned public detector, no private weights):
   `python scripts/fetch_detector.py --output ./detector` then the `--localize-only` command in "Public plate detector" below. This returns plate boxes, not characters.
2. **Character reading on real photos: you train your own model from data you are allowed to use.** Nothing here fetches a character dataset or distributes character weights (the maintainer's own trained weights are private). The steps:
   - Get a YOLO-format character dataset (`data.yaml`, one class per character using the names in `ARABIC_MAPPING` in `src/egypt_plate_ocr/core.py`). It can be your own labeled photos or a public one you have checked the terms of.
   - Train with `egypt_plate_ocr.training.train_local` (or the notebook's `OCR_RUN_TRAINING` settings), using `./detector/egyptian-plate-detector.pt` as the starting weights file, then read plates with `egypt-plate-ocr --image car.jpg --weights YOUR/best.pt`.
   - **Recorded original training data (public, terms not cleared by us):** the maintainer's private checkpoint was trained on the Roboflow dataset `egypt-plate-merged-erxeu` version 1, workspace `mousa-9wtzt`: https://universe.roboflow.com/mousa-9wtzt/egypt-plate-merged-erxeu/dataset/1 . Re-verified on 25 Nov 2026: the page lists license CC BY 4.0; version 1 (generated 20 Dec 2024) has 7,394 images (train 6,645 / valid 341 / test 408, no augmentations) and downloads in YOLOv8 format with a free Roboflow account. Its 36 class names are exactly the 34 names in `ARABIC_MAPPING` plus `car` and `License Plate`, which the decoder ignores by design, so its `data.yaml` works with this repository unchanged. Use version 1 to match the recorded training data; versions 2 and 3 are later, larger re-exports. Caveats you accept yourself: the CC BY 4.0 label is the uploader's, CC BY 4.0 requires attribution, image hashing with sampled visual checks found its photos largely come from the EALPR Vehicles research set whose original collection terms are not clearly established, and the same vehicles appear across its splits, so a score on its own valid/test split is optimistic (`docs/provenance.md`).
   - **Proven hosted run record:** a bounded free-CPU Kaggle run trained on the preserved export of this dataset and completed the full train_local to CLI cycle; the working hosted environment, observed results and limits are recorded in `docs/kaggle-real-data-verification.md`.
   - **Public alternative (not endorsed, not rights-cleared):** the project "egyptian car plates" by alyalsayed has 38 character classes (digits and Arabic letters). Sources: Roboflow Universe https://universe.roboflow.com/alyalsayed/egyptian-car-plates (as read on 25 Nov 2026 it lists license CC BY 4.0, 3 versions: v6 with 982 images, v12 with 3,724, v13 with 6,332 at 220x220) and Kaggle https://www.kaggle.com/datasets/alyalsayed/egyptian-car-plates . Downloading needs your own free Roboflow or Kaggle account, and this repository does not fetch or redistribute it. Open questions you must settle yourself before relying on it: the Kaggle page metadata was recorded as MIT while the archive's own README said CC BY 4.0 and Roboflow version 12 (`docs/split-leakage-audit.md`); the creator's project https://github.com/alyalsayed/Graduation_Project describes collecting photos from Instagram, so a dataset license label does not show that the underlying photos are cleared; the same plates appear across train and dev images (`docs/split-leakage-audit.md`), so a dev score is optimistic. We have not verified that this dataset matches the maintainer's private checkpoint, and nothing here claims it does.
3. **Synthetic mechanics demo (NOT evidence for real photos).** `scripts/yolo_synthetic_train.py` renders fictional plates, exports them in YOLO format, trains with the package adapter and writes `best.pt` that `egypt-plate-ocr` can load. It proves the generate, train and read plumbing runs end to end. It does not show the result reads real plates; the earlier synthetic-to-real check read 3 of 21 characters (`docs/synthetic500.md`). One epoch takes about a minute on one CPU core, so use a GPU (`--device 0`) or leave it running for 20+ epochs:
   ```sh
   uv run --no-sync python scripts/synthetic_plate_cpu.py generate --data ./syn
   uv run --no-sync python scripts/fetch_detector.py --output ./detector
   uv run --no-sync python scripts/yolo_synthetic_train.py --data ./syn --base ./detector/egyptian-plate-detector.pt --output ./char-model --epochs 20 --batch 8
   uv run --no-sync egypt-plate-ocr --image ./syn/test/0450.jpg --weights ./char-model/train/weights/best.pt --confidence 0.1
   ```

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

## Public plate detector (localization only)

A verified public checkpoint can localize Egyptian plates from a fresh clone today; character recognition still needs your own trained weights (the owner's checkpoint is private, and no public character model with a cleared license was found; for training data see the public candidate and its caveats under "Fork-to-output paths"). `scripts/fetch_detector.py` downloads the pinned, MIT-licensed `omasayeddd/egyptian-license-plate-detector` (single class `license_plate`) with SHA-256 verification:

```sh
python scripts/fetch_detector.py --output ./detector
egypt-plate-ocr --image path/to/car.jpg --weights ./detector/egyptian-plate-detector.pt --confidence 0.25 --auto-crop --detector-label license_plate --localize-only
```

`--localize-only` returns plate boxes and detector confidences without character decoding. Verified on CPU with the repository's pinned ultralytics 8.3.0; see [commands, output and rights notes](docs/public-detector.md). This is localization evidence, not OCR accuracy.

For a full vehicle image, opt in to separate localization and recognition:

```sh
egypt-plate-ocr --image path/to/car.jpg --weights path/to/best.pt --confidence 0.2 --auto-crop
```

Auto-crop mode detects `License Plate` boxes, adds a fixed small margin, and predicts each crop independently. It returns a JSON list with `box` (pixel xyxy), detector `confidence`, `letters` and `numbers` for each plate. No detected plate returns `[]`, with no full-frame fallback. Every detected plate is returned, including possible false positives. Confidence is the plate detector score, not a probability that its text is correct.

Missing plate box. With `--auto-crop` (turned on automatically by `--annotate` and `--char-boxes`), `--no-plate` chooses what happens when no `License Plate` box is found: `report` (CLI default) returns one entry with `"plate_found": false`, `"box": null` and empty text, so the miss is visible; `whole-frame` returns one entry marked `"plate_found": false, "fallback": "whole_frame"` with text read from the whole image (it can include characters outside any plate, and `--annotate` prints a fallback banner); `empty` returns `[]` as before. When a plate box is found, only characters inside it are read, and the others are ignored.

Which mode to use depends on the input. For full vehicle photos use `--auto-crop`. For images that are already plate crops (no `License Plate` box to find) skip `--auto-crop` and read the whole image: `--auto-crop` returns `[]` there. On the 408-image test split with a 15-epoch full-train nano model, direct reading got 119 of 238 plate-crop-style images exactly right while `--auto-crop` got 3, and `--auto-crop` got 126 of 170 vehicle photos while direct got 45 (see `docs/kaggle-real-data-verification.md`; valid and test overlap the training photos, so treat these as in-sample-leaning).

An optional `--char-source frame` exists (characters from the full-frame pass instead of a re-run on the crop). The default stays `crop`. It was measured as worse than `crop` (64 vs 126 of 170) and is not recommended.

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
