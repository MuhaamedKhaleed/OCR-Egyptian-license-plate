# Evaluation evidence and limits

## What this pilot checks

Offline unit tests cover ordering, unknown labels, finite coordinates, missing files, confidence validation, CLI failures and ensuring each image triggers a fresh prediction. Fake-model tests verify adapter behavior, not model accuracy.

## Before reporting recognition accuracy

Record the dataset source/version and its license, number of plates and images, class mapping, train/validation/test split, and whether images of the same vehicle appear across splits. Keep the test set separate from tuning. Define exact full-plate match, character error rate, detection precision/recall and mAP separately. Record hardware, batch size and image dimensions for latency claims.

Measure full-plate results on the held-out set, retaining representative successes and failure cases without exposing identifying information unnecessarily. Include lighting, blur, angle and occlusion limitations. Do not infer plate accuracy from detection mAP or unit-test coverage.

The current training notebook has not been rerun. Its saved-output metrics were recorded before outputs were removed; they are historical observations, not newly reproduced benchmarks. No clinical, hardware, production or parking-system impact claims follow from this code cleanup.


## Bounded external CPU evaluation, 2026-11-24

Source: https://github.com/Abdalllah10Ahmed/ALPR-Egyptian-Plates
The repository includes 50 self-collected images for reproducibility and an MIT license. Tested the first 10 sorted filenames, `img_001.jpg` through `img_010.jpg`, without success-based selection. Source ground truth was visually reviewed against each plate; transliteration ambiguities were not accepted as exact Arabic matches.

Used the recovered original checkpoint with confidence 0.2, unchanged weights, and no training. Each full frame was tested, then its highest-confidence detected plate crop with a small fixed margin. Exact means the Arabic letters and digit sequence both match, including hamza; display spaces are separators.

- Full-frame exact text: 2 of 10.
- Detected-crop exact text: 6 of 10.
- Digit sequence correct: 9 of 10 full frames, 10 of 10 crops.
- Intended plate localized in all 10; one image also produced an extra plate detection. This is not a detection benchmark or precision estimate.

Failures included missing letters, ra/waaw confusion, wrong inherited Arabic aliases, hamza loss, taa/tah confusion and a residual duplicate character in one crop. The conservative IoU duplicate rule was fixed before this sample was evaluated; it was not tuned to these answers.

These convenience-sample fractions are NOT general accuracy estimates. The source repository dates from 2026, later than the saved April 2025 checkpoint, but source chronology alone cannot certify image novelty and the original training corpus was not exhaustively checked for overlap. No claim of independent held-out evaluation is made. No sample images or checkpoint copies are included here. Preserve the source license/attribution if redistributing its files. No model or repository license was changed.


## Annotation-grounded mapping correction

The original saved Kaggle dataset annotation cache and its matching training images were inspected for ambiguous aliases. Reviewed classes included `aain`/`e` (ع), `alef`/`alf` (أ), `daal`/`dal` (د), `dad` (ض), `taa` (ط), `waaw`/`waw` (و) and `zay` (ز). Original source: https://www.kaggle.com/code/mohamedkhaled07/model-ocr/output and https://universe.roboflow.com/mousa-9wtzt/egypt-plate-merged-erxeu . This is a bounded alias audit, not verification of every class or annotation. No training images, cache or checkpoint is redistributed.

Opt-in auto-crop now recognizes each detected plate separately, with a fixed margin clipped to image bounds. The first 10 evaluation images above became development evidence and are not retained as a held-out test.

## Prospective next-slice check, 2026-11-24

After freezing the revised mapping and crop code, ran sorted images `img_011.jpg` through `img_020.jpg` from the same external source. Same original checkpoint, confidence 0.2, CPU only. No retraining or threshold tuning. The source ground-truth list had been visible during earlier repository inspection, so this is prospective execution, not blinded evaluation. Reviewed the actual plate pixels after inference.

- Exact text on the highest-confidence primary plate: 7 of 10.
- Correct primary digit sequence: 9 of 10.
- Intended plate localized in all 10, with an additional background-plate output in image 017. This is not full-scene exact-match scoring or a detection benchmark.
- Failures: image 016 confuses kaf/faa and duplicates a digit 1; images 017 and 019 omit dal. The extra background output in 017 has incomplete text.

Code was not retuned to these results. Different image slices prevent treating 6/10 versus 7/10 as a controlled improvement. This small same-source slice is not a general accuracy estimate; original training overlap is uncertified. The three failures are unresolved. A larger independently held-out and vehicle-separated evaluation is still needed.
