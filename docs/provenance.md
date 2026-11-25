# Provenance and rights evidence, 24 November 2026

This is an evidence inventory, not legal clearance or a claim of ownership.

- The owner clarified the inherited credential/code came from someone on Kaggle. The exact upstream authorship chain for every notebook cell is not established. The new MIT notice applies only to owned engineering work; it does not relicense inherited material.
- The previously identified upstream project is https://github.com/Abdalllah10Ahmed/ALPR-Egyptian-Plates . A similar project or matching name does not by itself establish this notebook's origin or a grant for every asset.
- Ultralytics 8.3.0's installed distribution includes AGPL-3.0 license text. Official vendor guidance is https://www.ultralytics.com/license and the license text is https://www.ultralytics.com/legal/agpl-3-0-software-license . Vendor guidance describes open-source/Enterprise alternatives and trained-model conditions. Do not assume scoped MIT clears a larger deployment. No paid license was bought.
- A Roboflow dataset license label previously observed as CC BY 4.0 is not enough to clear the merged dataset's source images, all annotations or the recovered checkpoint. Record the exact dataset version, source/attribution and weight training lineage before redistribution. No such clearance is asserted.
- The local one-epoch smoke uses generated rectangles/random initialization. It makes no real-plate quality claim and distributes no detector weights. The original private recovered checkpoint was not needed for this training smoke.

Pending evidence: actual inherited code source/license, complete dataset composition and attribution, compatible weight provenance/training terms, and rights for any future published photos. These affect redistribution and deployment; they do not block default decoder tests or local engineering work.

## Addendum, 25 November 2026: original training dataset identified

- The recovered private checkpoint's recorded training data is Roboflow dataset `egypt-plate-merged-erxeu` version 1, workspace `mousa-9wtzt` (https://universe.roboflow.com/mousa-9wtzt/egypt-plate-merged-erxeu/dataset/1). Its exported `data.yaml` (preserved from the original notebook workspace) records `license: CC BY 4.0`, 36 class names, and the Roboflow project URL/version. The 36 names are the 34 names in `ARABIC_MAPPING` (`src/egypt_plate_ocr/core.py`) plus `car` and `License Plate`, which the decoder ignores by design.
- Re-verified live on 25 Nov 2026: the Universe page still lists license CC BY 4.0; version 1 (generated 20 Dec 2024) holds 7,394 images (train 6,645 / valid 341 / test 408, no augmentations) with YOLOv8-format download behind a free Roboflow account. Versions 2 (7,587) and 3 (7,639) are later re-exports and do not match the recorded training data.
- Source overlap: image hashing (8x8 dHash, Hamming <= 6, including mirrored) matched 2,084 of 2,087 EALPR Vehicles images to the version 1 export, with same-photo pairs confirmed on visually inspected montage samples. So the merged dataset's photos largely derive from the EALPR Vehicles research set (https://github.com/ahmedramadan96/EALPR), whose original collection and redistribution terms are not clearly established. The CC BY 4.0 label is the Roboflow uploader's and does not settle the underlying photos' rights. No clearance is asserted.
- Split independence: matched images appear across the export's train, valid and test splits (nearest matches: train 1,925 / valid 102 / test 57), so any score measured on its own valid/test split is in-sample-leaning, not held-out.
