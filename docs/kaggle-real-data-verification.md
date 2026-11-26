# Hosted real-data training verification (26 November 2026)

Bounded verification that `egypt_plate_ocr.training.train_local` trains on the original training data export and that the resulting checkpoint runs through the package CLI, on a free hosted CPU runner. No new accounts, paid compute, weight publication or recognition-quality claim.

## Source data

The preserved YOLO export of `egypt-plate-merged-erxeu` version 1 (Roboflow, CC BY 4.0, 7,394 images, 36 classes; provenance in README), mounted as a Kaggle notebook input from the original `model_ocr` notebook output. No fresh download and no redistribution.

## Proven hosted environment

Kaggle's current base image ships Python 3.13.18 with a numpy 1.26.4 build compiled against numpy-2 headers; importing it fails, and numpy<2 publishes no cp313 wheels, so the base interpreter cannot run this stack. The verified route is a uv-managed Python 3.11.18 virtual environment:

- Python 3.11.18 (uv-managed), numpy 1.26.4, pandas 3.0.6, torch 2.4.1+cpu, torchvision 0.19.1+cpu, opencv-python 4.11.0.86, ultralytics 8.3.0 (its own resolved dependency set), this repository at commit 5757b354.
- `MPLBACKEND=Agg` is required: the base image exports a matplotlib-inline backend that a clean virtual environment does not provide.
- The base image preinstalls wandb, and the Ultralytics automatic wandb callback then fails `model.train` with `UsageError: project '/kaggle/working' invalid`. The clean venv has no wandb and does not hit this.

## Observed run

Three CPU epochs trained from the documented detector weights (sha256 c8f4fbee9040cf316a2bf3649689d55db8a845f83151d85c8b82a1525a5690a6, verified before training) over the preserved export. Classification loss fell each epoch (4.94 to 3.78) and was still falling at stop; final validation mAP50 was 0.057 with recall 0.085. The full train_local to best.pt to CLI cycle completed with exit code 0. The CLI (`--auto-crop`, confidence 0.1) produced a valid empty result (`[]`) for the single test photo (`1138FF`, 640x640, ground truth letters ف ف, numbers ٨ ٣ ١ ١). This run proves the wiring only. It does not show why the output was empty: the log has no raw detections, so undertraining, the plate gate not firing at confidence 0.1, and the crop re-run described below are all untested explanations for this specific run. Three epochs also fell entirely inside Ultralytics' default 3-epoch warmup (`warmup_epochs=3.0`).

## Follow-up diagnostic (26 November 2026, same environment, CPU)

Label statistics on the train split (6,645 label files): 55% (3,680) have no `License Plate` box; the median character box is 57 x 152 px at 640 (so a large share of the data is plate-crop style); all sampled images are 640x640; class counts are very uneven (`dad` 1, `0` 2, `zay` 9, `e` 14, `kaaf` 30 instances). `License Plate` and `car` are the only classes the decoder ignores; the other 34 map.

Controlled memorization test (not a held-out result): 32 training images, 120 epochs, 480 steps, `warmup_epochs=0`, no augmentation, from the same detector weights. Train-set mAP50 reached 0.964, so the head, label mapping and decode order work. With the same weights on the same 32 images at confidence 0.25:

| Decode path | Exact plate text |
| --- | --- |
| Whole image, character boxes only | 29 / 32 |
| `--auto-crop` (plate box, then re-run on the crop) | 4 / 32 |
| `--auto-crop --char-source frame` | measured next, see below |

The `License Plate` gate fired on 32 / 32 images, so the gap comes from re-running the model on a tight crop, where characters appear at a scale the model was not trained on. An earlier 80-epoch run (320 steps) did not memorize the same set, which shows the step count matters for such a test. At confidence 0.001 both paths read 0 / 32 (about 105 raw boxes per image of noise). This result is limited to this small training-set test.

`--char-source frame` (new, opt-in; the default stays `crop`) takes the characters from the one full-frame pass and keeps those whose box center lies inside each plate box.

## Weights and cleanup

The first run saved `best.pt` and `last.pt` into the private version output of a temporary Kaggle notebook, which was a mistake against the no-uploaded-weights rule. The notebook (and with it all seven version outputs) was then deleted and its URL returned "page not found". Kaggle's backend purge timing cannot be checked. The weights were never public and were never committed here. The original `model_ocr` notebook and its preserved export output are unchanged. The follow-up diagnostic ran in an interactive draft session with persistence off and wrote its weights only under `/tmp`; no notebook version was saved.
