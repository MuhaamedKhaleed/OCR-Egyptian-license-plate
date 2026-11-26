# Egyptian License Plate OCR

YOLO-based reading of Egyptian license plates: a public plate detector for localization plus a character-recognition model you train yourself, with a small tested decoding package and CLI. Output is UTF-8 JSON with `letters` and `numbers`, plus optional per-character box annotation.

## Results (honest)

- **Plate localization works from a fresh clone today** using a pinned, hash-checked public detector (`omasayeddd/egyptian-license-plate-detector`, MIT): see the quickstart. This returns plate boxes, not characters.
- **Character reading requires training your own model** on data you are allowed to use. The maintainer's original trained checkpoint works with this code but is private and not published.
- A recorded 15-epoch training run (free Kaggle CPU/GPU, `docs/kaggle-real-data-verification.md`) read 119/238 plate-crop-style test images exactly right in direct mode, and 126/170 full vehicle photos in `--auto-crop` mode. **Valid/test images overlap the training photos, so these are in-sample-leaning numbers, not held-out OCR accuracy.** No held-out plate-level accuracy has been measured.
- The pre-cleanup notebook reported detection validation mAP@0.5 ~0.886 and mAP@0.5:0.95 ~0.543 on 341 images. These were not rerun and are detection metrics, not recognition accuracy. No "94% recognition" claim is made.
- Offline package tests and CPU inference with the original checkpoint are verified; a synthetic-plates plumbing demo trains and reads end to end but does not read real plates (3/21 characters synthetic-to-real, `docs/synthetic500.md`).

## Quickstart

Needs Python 3.11 or 3.12 (`uv sync --python 3.11 --locked --extra inference --extra dev`), ~2 GB RAM, internet for the first install/downloads.

**1. Localize plates (works immediately, no private weights):**

```sh
python scripts/fetch_detector.py --output ./detector
egypt-plate-ocr --image car.jpg --weights ./detector/egyptian-plate-detector.pt --confidence 0.25 --auto-crop --detector-label license_plate --localize-only
```

**2. Read characters: train your own model.** Get a YOLO-format character dataset (one class per character, using the names in `ARABIC_MAPPING` in `src/egypt_plate_ocr/core.py`), train with `egypt_plate_ocr.training.train_local`, then:

```sh
egypt-plate-ocr --image car.jpg --weights YOUR/best.pt --confidence 0.2 --auto-crop
```

The recorded training data for the maintainer's checkpoint is the public Roboflow dataset `egypt-plate-merged-erxeu` v1 (7,394 images, page lists CC BY 4.0, classes match this code). Caveats you accept yourself: CC BY 4.0 requires attribution, the photos appear to come largely from a research set whose original collection terms are not clearly established, and the same vehicles appear across splits (`docs/provenance.md`). Download needs a free Roboflow account; this repository fetches and redistributes no dataset.

Modes: for full vehicle photos use `--auto-crop`; for images that are already plate crops, skip it. `--no-plate report|whole-frame|empty` controls behavior when no plate box is found. `--tta` and `--char-boxes` (per-character boxes inside the plate box) are optional extras. `model-ocr.ipynb` is an opt-in local workflow with no install/download cells; training runs only when `OCR_RUN_TRAINING=1` is set.

## Limits

- No published weights, no fetched dataset, no hosted demo. Single-crop mode does not establish legal plate validity; every detected plate box is returned, including false positives.
- Arabic aliases were reviewed against original training annotations, but recognition errors remain.
- Verified on Linux; the CI container is a CLI smoke image without Ultralytics or weights and does not validate real OCR inference. Free hosted CI only, no published artifacts.
- Ultralytics 8.3.0 is AGPL-3.0 with its own model-licensing conditions; this repository's MIT notice does not override them (`docs/provenance.md`).

## Security note

A third-party Roboflow API key was previously committed here (the maintainer had copied the code from someone else's public notebook; the key was never his and was never used or revoked by this project). It was **removed from the entire git history on 26 Nov 2026**; the history was rewritten and force-pushed, so old clones and forks diverge and should be re-cloned. External copies outside this repository (an old Kaggle notebook) are handled separately. Never commit a replacement key; the package needs no credential.

## License

MIT for the project-owned code (see `LICENSE` and `LICENSE-SCOPE.md`). Historical notebooks, datasets, pretrained checkpoints and dependencies keep their own terms; no rights clearance is claimed for them. Details and evidence: `docs/verification.md`, `docs/evaluation.md`, `docs/provenance.md`, `CHANGELOG.md`.
