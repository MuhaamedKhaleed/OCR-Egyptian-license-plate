# Evaluation evidence still needed

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
