# Per-character boxes

`egypt-plate-ocr --auto-crop --char-boxes` adds a `characters` list to each plate. Each entry is the
model's own detection: `char` (glyph read), `label` (class name), `confidence`, and `box` (xyxy in
the plate crop; add `box[0]`, `box[1]` of the plate for full-image pixels). Nothing is estimated.

`--annotate out.jpg` also writes the image with a black plate box and one coloured box per character,
with the read glyph under it. Both flags need local trusted weights with character classes.

```
egypt-plate-ocr --image car.jpg --weights best.pt --auto-crop --annotate out.jpg
```

Status: unit tests cover the box output and the renderer on synthetic data. The CLI path was also run
end to end on a real model: Kaggle notebook, Ultralytics 8.3.0, the original (private, not published)
yolov8m checkpoint, full-size Pexels photos, exit code 0 and per-character boxes drawn on the image.
That was a small, clear, already-seen sample, so it shows the feature runs, not how accurate it is.
Results depend on image scale: the same photo resized differently gave different text. The label under
each box is the Arabic glyph when an Arabic-capable font is found (Noto, DejaVu, or the DejaVu file
that ships with matplotlib); otherwise the class name (for example qaf or 2) is drawn.

# Training options

`train_local(..., batch=, device=, seed=, amp=)` now accepts GPU settings. Defaults stay CPU, batch 1,
amp off. A direct Ultralytics 8.3.0 run on a T4 used: yolov8m.pt, 15 epochs, batch 16, imgsz 640,
seed 0, deterministic, workers 2, amp on. Valid mAP50 / mAP50-95: 0.9015 / 0.5555 vs the old
checkpoint's 0.8897 / 0.5471 (about the same; detection metrics, not text accuracy). That run did not
go through `train_local`, so `train_local` on GPU is untested. Weights are never committed.
