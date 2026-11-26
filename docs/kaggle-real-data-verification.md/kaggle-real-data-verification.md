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

Three CPU epochs trained from the documented detector weights (sha256 c8f4fbee9040cf316a2bf3649689d55db8a845f83151d85c8b82a1525a5690a6, verified before training) over the preserved export. Classification loss fell each epoch (4.94 to 3.78) and was still falling at stop; final validation mAP50 was 0.057 with recall 0.085. The full train_local to best.pt to CLI cycle completed with exit code 0. The CLI produced a valid empty result (`[]`) and an unchanged annotated image for the single test plate photo (ground truth: letters ف ف, numbers ٨ ٣ ١ ١). Three CPU epochs are far too few for usable detections, so empty output here is expected undertraining rather than evidence of a pipeline defect; equally, this run is wiring evidence only and claims no recognition quality.

## Weights and cleanup

The temporary private verification notebook, including its run weights, was deleted after evidence recovery; no trained weights are committed here or retained in hosted outputs. The original `model_ocr` notebook and its preserved export output are unchanged.
