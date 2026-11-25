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

## Prospective third slice and nested-duplicate fix, 2026-11-25

Sorted images `img_021.jpg` through `img_050.jpg` from the same external source, same recovered checkpoint, confidence 0.2, CPU only. Every source ground-truth label in this slice was re-verified visually against the plate pixels before scoring; all 30 matched. The ground-truth list had been visible during earlier repository inspection, so this remains prospective execution on a same-source convenience sample, not a blinded or vehicle-separated held-out test. Training-corpus overlap is uncertified.

- Primary-crop exact text: 27 of 30. Correct digit sequence: 28 of 30. Localized: 30 of 30, no extra plate detections in this slice.
- Hard conditions present and read exactly: motion blur (042, 049), a cable across the plate (047), weathered/faded plates (040, 050), tilted angle (031). Low-light 033 failed.
- Failures: 033 and 037 are physically reversed-order digit plates (digits stamped right-to-left, unlike the other 48); the fixed right-to-left center sort inverts the number, and 033 also added a nested duplicate digit. 048 fired `faa` (0.39) beside the correct `kaaf` on the same glyph below the 0.8 IoU threshold.

Decoder duplicate handling was then extended: nested same-label boxes (one covering at least 0.8 of the other) now keep only the strongest detection. Different labels on the same glyph are deliberately kept, because resolving them by confidence alone picks the wrong class on 048. Measured on the cached raw detections of all 50 source images with only the decoder changed: primary-crop exact 43 to 44 (008 duplicate `alef` fixed; nested duplicate digits in 016 and 033 removed without making either exact), no regressions.

Full 50-image source corpus with current code: localized 50 of 50, primary-crop digit sequence 48 of 50, primary-crop exact 44 of 50, full-frame exact 17 of 50. Remaining failures: 016 `kaf` read as `faa`, 017 and 019 omit `dal`, 033 and 037 reversed digit order, 048 spurious `faa`. Auto-crop remains essential: full-frame decoding without it is far weaker. These are same-source convenience-sample counts, not general accuracy estimates.

Real-weights integration is now repeatable: `scripts/evaluate_external_slice.py` reruns this evaluation locally, and the installed CLI (`--auto-crop` and single-crop modes) was exercised against the recovered checkpoint with exit status 0 and correct output. CI still excludes weights, so this integration remains a documented local procedure, not an automated CI check.
