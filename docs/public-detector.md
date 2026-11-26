# Public pretrained plate detector route

25 November 2026. Verified localization route for someone cloning this
repository without the owner's private checkpoint.

## What exists and what does not

- Public, license-checked weights for **plate localization**:
  `omasayeddd/egyptian-license-plate-detector` (Hugging Face), MIT per its
  model card front matter and the HF API license field at fetch time. Single
  class `license_plate`; the card states YOLOv8n trained on the EALPR
  Egyptian Vehicles dataset and reports validation mAP@50 0.995 /
  mAP@50-95 0.918 (not re-measured here).
- No qualified public weights for the **character recognition** stage were
  found. `sshdopey/egyptian-license-plates` carries no license and was
  excluded. The owner's original checkpoint remains private, so a fresh
  clone can localize plates but cannot read characters until the owner
  publishes compatible weights or the user trains their own.
- The rejected candidate and the private-checkpoint status were checked
  25 November 2026 via the HF API.

## Verified commands (run 25 November 2026, CPU)

```sh
uv sync --locked --extra inference
uv run --no-sync python scripts/fetch_detector.py --output ./detector
uv run --no-sync egypt-plate-ocr --image path/to/car.jpg \
  --weights ./detector/egyptian-plate-detector.pt \
  --confidence 0.25 --auto-crop --detector-label license_plate --localize-only
```

`scripts/fetch_detector.py` pins revision
`b9c1ace323f96988f0ee58106a7942b05536c007` and SHA-256
`c8f4fbee9040cf316a2bf3649689d55db8a845f83151d85c8b82a1525a5690a6` for
`best.pt`; the served bytes matched the `x-linked-etag` Hugging Face
returned at that revision. The detector's `config.json` is pinned the same
way.

## Actual output

Three vehicle images from the EALPR Vehicles folder
(`0001.jpg`, `0100.jpg`, `1000.jpg`, raw files from
github.com/ahmedramadan96/EALPR, path `EALPR Vechicles dataset/Vehicles/NNNN.jpg`, spelled that way upstream), ultralytics 8.3.0, torch 2.4.1+cpu,
python 3.12:

```json
[{"box": [328, 368, 449, 433], "confidence": 0.9110252261161804}, {"box": [666, 208, 707, 233], "confidence": 0.4201603829860687}]
[{"box": [437, 507, 753, 677], "confidence": 0.9049633145332336}]
[{"box": [232, 614, 372, 686], "confidence": 0.9023663401603699}]
```

Each image's main plate was found; the second low-confidence box on the
first image may be a second vehicle or a false positive and was not
visually adjudicated here. These images come from the dataset the card
says the detector trained on, so this run verifies the download-to-output
route, not generalization.

## Options added

- `--detector-label` names the plate class inside the supplied weights
  (default `License Plate`; the public detector uses `license_plate`).
- `--localize-only` (requires `--auto-crop`) returns boxes and detector
  confidences without character decoding, for weights with no character
  classes.

Pairing public localization weights with separate character weights in one
run is not supported; the CLI still takes a single weights file.

## Rights notes

- Model: MIT per model card. Ultralytics stays AGPL-3.0 software; the
  card's MIT tag is the uploader's claim and does not settle obligations
  that may attach to weights produced with Ultralytics. See
  `docs/provenance.md`.
- Dataset: the EALPR GitHub repository has no LICENSE file (README
  citation request only); the Kaggle re-upload labels the Vehicles subset
  Apache-2.0. Treat dataset-level rights as unverified.
- `best.pt` loads with the repository's pinned ultralytics 8.3.0.
