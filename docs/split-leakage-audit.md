# Read-only cross-split duplicate screening

This tool screens explicitly named image folders before model training. It does not prove independent vehicles, capture sessions, authenticity or correct labels. A zero count is not a clean bill of health.

Byte-only mode uses the standard library. It catches only identical file bytes, not recoding or small transforms:

    python scripts/audit_image_splits.py --split train=LOCAL_TRAIN --split dev=LOCAL_DEV --output NEW_AUDIT.json

Optional visual mode requires a local Pillow installation. It hashes EXIF-oriented RGB pixels including image dimensions and calculates a 256-bit horizontal difference hash after bilinear resizing to 17x16. It catches identical decoded pixels and flags perceptual candidates by Hamming distance. Default threshold16 is a screening setting, not a validated duplicate decision threshold:

    python scripts/audit_image_splits.py --split train=LOCAL_TRAIN --split dev=LOCAL_DEV --visual --threshold 16 --output NEW_VISUAL_AUDIT.json

Only supplied folders are inspected, recursively. Do not supply a heldout test folder before you intend to expose it. The tool never auto-discovers test, imports labels, drops rows, groups images or deletes files. Split directories must be distinct and nonnested; symlink image files are rejected. Unreadable images are listed as errors. Visual mode requires decodable images; byte-only mode cannot detect corrupt image bytes. If some reads fail, CLI saves the partial report with complete_read=false and exits2. No readable files in a split fails without output. File names in the report can contain private IDs, so keep it private.

Cross-split comparisons are exhaustive over readable inputs, capped at5million pairs by default; exceeding the cap fails explicitly rather than sampling. Findings are classified mutually exclusively as same_bytes, same_pixels or perceptual_candidate. Counts cover every pair while the saved list is capped at10000findings; findings_truncated records that cap. A count is pairs, not unique images. Candidate pairs need visual review. Uniform backgrounds and repeated layouts can be false positives. Rotations, crops, larger changes and semantically similar views can be missed. Within-split duplicates are outside this tool's scope.

## Tests and actual source check

Fifteen new tests use fictional inputs:12standard-library tests and3optional Pillow tests. They cover unchanged inputs, no-overwrite output, split boundaries, count/list caps, unsafe paths, same-pixel/different-encoding images, corrupt inputs and a known perceptual false-positive case. The latter intentionally shows why candidate does not mean duplicate. No real registration fixture is committed.

Ran the tool on already-inspected train/dev folders of https://www.kaggle.com/datasets/alyalsayed/egyptian-car-plates/data .2608train and745dev images made1942960cross-split comparisons. No identical bytes/pixels,10perceptual candidate pairs at threshold16 with this exact bilinear implementation. This is separate from the preliminary nearest-neighbor screen using another resize filter, so its count is not interchangeable. The preliminary top12pairs were visually checked and showed the same registration, view and background with small transforms. No independent split or real-world accuracy claim follows. The archive also contains full-alphabet character classes and pasted-looking glyphs with uncertain origins. No model trained or heldout test pixels/labels read from that source.

Kaggle metadata says MIT while the actual archive README.dataset.txt specifies CC BY4.0 and Roboflow version12. Preserve source attribution and do not treat metadata as proof all source photos have cleared rights. Creator project https://github.com/alyalsayed/Graduation_Project describes Instagram collection; original source https://universe.roboflow.com/alyalsayed/egyptian-car-plates . Source data, audit pair paths, photos and registration labels remain local, not bundled or published.

New owned screening code is MIT; external data rights separate. Review branch only. No model score increase or external fees.
