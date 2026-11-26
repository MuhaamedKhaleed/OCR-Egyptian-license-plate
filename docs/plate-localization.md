# Real-image plate localization, 24 November 2026

This is an optional detector experiment, not full text recognition. Egyptian Car Plates version13: https://www.kaggle.com/datasets/mahmoudkhater99/car-plate . The page describes self-collected images and declares Database ODbL / Contents DbCL. Source has plate bounding boxes, not plate text. Terms: http://opendatacommons.org/licenses/dbcl/1.0/ and https://opendatacommons.org/licenses/odbl/1-0/ . Preserve attribution and applicable share-alike/alteration-method obligations. Code MIT does not relicense third-party code, data or weights. Ultralytics terms: https://www.ultralytics.com/license . No raw images or weights published.

## Exact measured scope

Sorted XML files; retain valid category1 boxes, exact-pixel deduplicate, seed42shuffle, take96. First16dev (19plates), remaining80train. Related original capture sessions, not scene-independent test. Other category labels excluded, no fake text labels. All16dev predictions inspected against ground truth internally, visible real localization with remaining misses. Images are not redistributed as a report illustration.

First15epoch run on images precompressed to320pixels: mAP50 .01418, precision .00470, recall .89474. Many plates were only10to30pixels wide after compression. Rebuilding originals at640and evaluating the same checkpoint gave mAP50 .0617, still poor. Four more epochs with640original-resolution data, batch1, backbone freeze10 and no mosaic reached mAP50 .73754, mAP50-95 .43818, precision .83686, recall .63158. This is combined preprocessing/training/freeze change, not a controlled single-bug attribution. First640batch8 attempt exhausted memory; batch1 succeeded. Refined run repeated independently with identical metrics. Then the entire portable pipeline, including dataset selection and both training stages, reproduced both earlier and refined metrics exactly.

These are development-selected numbers on reused16images. No untouched scene-level test, character transcriptions, full-plate recognition, field performance, fairness or production guarantee. Do not present .7375 as OCR accuracy.

## Reproduce explicitly on CPU

Obtain source version13 and official YOLOv8n COCO checkpoint locally, respecting their terms. Dataset acquisition is separate and not automated. The script requires paths and a new output directory; it does no work on import and does not intentionally download anything. Do not supply an untrusted pickle checkpoint.

```sh
uv sync --locked --extra dev --extra inference
uv run --no-sync python scripts/plate_localization_cpu.py --data LOCAL_VERSION13 --weights LOCAL_OFFICIAL_YOLOV8N_PT --output NEW_OUTPUT --stage prepare
uv run --no-sync python scripts/plate_localization_cpu.py --data LOCAL_VERSION13 --weights LOCAL_OFFICIAL_YOLOV8N_PT --output NEW_OUTPUT --stage initial
uv run --no-sync python scripts/plate_localization_cpu.py --data LOCAL_VERSION13 --weights LOCAL_OFFICIAL_YOLOV8N_PT --output NEW_OUTPUT --stage refine
```

Use the same output path for the three ordered stages. Each stage refuses existing training results rather than replacing them. Default --stage all runs all three uninterrupted, about4minutes CPU including reading original photos. Stages allow execution under bounded command timeouts. Preparing creates a local source manifest and train/dev files; training creates local weights, keep these out of GitHub. Fixed seed/environment reproducibility is not cross-platform bitwise determinism.

## Measured portable run

```json
{
  "source": "https://www.kaggle.com/datasets/mahmoudkhater99/car-plate",
  "source_version": 13,
  "stage1": {
    "metrics/precision(B)": 0.004703929164360819,
    "metrics/recall(B)": 0.8947368421052632,
    "metrics/mAP50(B)": 0.014175760439060357,
    "metrics/mAP50-95(B)": 0.007132366080810093,
    "fitness": 0.00783670551663512
  },
  "stage2": {
    "metrics/precision(B)": 0.8368649231911353,
    "metrics/recall(B)": 0.631578947368421,
    "metrics/mAP50(B)": 0.7375414550398431,
    "metrics/mAP50-95(B)": 0.4381757463191288,
    "fitness": 0.4681123171912003
  },
  "seconds": 73.01271033287048,
  "limits": "Development only80train16dev from related captures, same reused dev selection. Not scene-independent or text OCR. Source DbCL/ODbL and Ultralytics terms separate. No raw plates or weights redistribution."
}
```

## Frozen-model additional check

After fixing the refined checkpoint, evaluated the first32 sorted previously unused annotated original images (41plates), excluding the96selected filenames and exact-pixel hashes. No training or tuning on these32. mAP50 .73694, mAP50-95 .38952, precision .81268, recall .63415. These are new images to this experiment, but still related captures from the same original collection, not a scene-disjoint or external benchmark. Once inspected, this additional set is exposed and must not be called an untouched test for future work.
