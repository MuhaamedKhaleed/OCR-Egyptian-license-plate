# Evaluation evidence still needed

## What this pilot checks

Offline unit tests cover ordering, unknown labels, finite coordinates, missing files, confidence validation, CLI failures and ensuring each image triggers a fresh prediction. Fake-model tests verify adapter behavior, not model accuracy.

## Before reporting recognition accuracy

Record the dataset source/version and its license, number of plates and images, class mapping, train/validation/test split, and whether images of the same vehicle appear across splits. Keep the test set separate from tuning. Define exact full-plate match, character error rate, detection precision/recall and mAP separately. Record hardware, batch size and image dimensions for latency claims.

Measure full-plate results on the held-out set, retaining representative successes and failure cases without exposing identifying information unnecessarily. Include lighting, blur, angle and occlusion limitations. Do not infer plate accuracy from detection mAP or unit-test coverage.

The current training notebook has not been rerun. Its saved-output metrics were recorded before outputs were removed; they are historical observations, not newly reproduced benchmarks. No clinical, hardware, production or parking-system impact claims follow from this code cleanup.
