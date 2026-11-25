# Optional test-time augmentation (`--tta`)

`--tta` (with `--auto-crop`) passes `augment=True` to Ultralytics for the character pass on each plate crop. It is off by default, slower, and not a proven accuracy gain.

What was measured (same inputs, one private weights file SHA256 `a715da45304aad5be4705d21e1ad9544b1e2cf34e4588134879514a033657f54`, Ultralytics 8.4.218, CPU, conf 0.2): 8 clear Pexels plate photos, exact match of letters and digits against a hand read of the plate, in a resized copy (2000 px wide) and the full-size original. This is the same sample used to pick the setting, so it is not an independent accuracy figure.

| Setting | Resized | Original |
| --- | --- | --- |
| Default (crop imgsz 640) | 6/8 | 6/8 |
| Crop imgsz 320 | 8/8 | 7/8 |
| Crop imgsz 480 / 960 / 1280 | 6 / 7 / 6 | 6 / 6 / 6 |
| Vote over 5 crop scales | 7/8 | 6/8 |
| Plate detect at imgsz 1280 | 7/8 | 6/8 |
| `--tta` | 8/8 | 7/8 |

Known failure that stays: one full-size photo (4160x6240) gets no plate detection at default size, with or without `--tta`. Detecting the plate at imgsz 1280 finds it but adds a spurious digit.

Use `--tta` as an experiment. Verify on your own clear photos before relying on it.
