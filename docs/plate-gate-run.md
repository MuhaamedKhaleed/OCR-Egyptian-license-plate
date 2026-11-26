# Plate box gate run

This is a separate training run from the one in `docs/kaggle-real-data-verification.md` (not the same checkpoint, so its counts are not comparable with that document and must not be mixed with it). Setup: nano public plate detector as the start, full train split (6,645 images), 15 epochs, free Kaggle T4, pinned ultralytics 8.3.0. Weights stayed in `/tmp` of an interactive session with persistence off; no version was saved and the temporary notebook was deleted afterwards (its URL returns "page not found"). The test split overlaps the training photos (`docs/provenance.md`), so these numbers lean in-sample. The evaluation script was written for this check, with ground truth decoded from the label files, at confidence 0.25 on the 408 test images.

Behaviour tested: `--auto-crop` (implied by `--annotate` and `--char-boxes`) reads characters only inside the found plate box. When no plate box is found, `--no-plate report` (CLI default) returns one entry with `plate_found: false` and no text, `--no-plate whole-frame` reads the whole frame and marks the entry `fallback: "whole_frame"`, and `--no-plate empty` returns `[]`.

| Subset | Images | Plate found | Exact, plate box gate only | Exact, with marked whole-frame fallback | Exact, direct (no gate) |
| --- | --- | --- | --- | --- | --- |
| All | 408 | 172 | 129 | 300 | 238 |
| Vehicle photos (have a plate box label) | 170 | 169 | 126 | 127 | 65 |
| Plate-crop style (no plate box label) | 238 | 3 | 3 | 173 | 173 |

On the vehicle photos the gate was right and direct reading wrong on 68 images, and the reverse on 6. On plate-crop style images the gate almost never fires, so the default `report` mode returns no text there. Use `--no-plate whole-frame` (visibly marked in the JSON and on the annotated image) or no `--auto-crop` for inputs that are already plate crops.

The earlier run and its different numbers are in `docs/kaggle-real-data-verification.md`.
