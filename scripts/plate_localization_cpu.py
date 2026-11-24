"""Explicit local Egyptian plate localization experiment. Not text recognition.

Obtain source version13 yourself, accept its terms, and supply official
Ultralytics YOLOv8n COCO weights locally. No automatic download is intended.
Outputs include trained weights and source manifest; keep those local.
"""


def main():
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--stage", choices=["prepare", "initial", "refine", "all"], default="all")
    args = parser.parse_args()
    if args.stage in {"prepare", "all"} and args.output.exists():
        parser.error("Output exists; choose a new directory.")
    if not args.weights.is_file():
        parser.error("Supply a local official YOLOv8n COCO checkpoint.")
    if not (args.data / "annotations").is_dir() or not (args.data / "Pics").is_dir():
        parser.error("Expected source version13 with annotations/ and Pics/.")

    if args.stage in {"initial", "refine"}:
        stage_dir = "stage1" if args.stage == "initial" else "stage2"
        if (args.output / stage_dir).exists():
            parser.error("Training stage output exists; choose a fresh prepared directory.")
        if not (args.output / "data640/data.yaml").is_file():
            parser.error("Run prepare first.")
        if args.stage == "refine" and not (args.output / "stage1/weights/best.pt").is_file():
            parser.error("Run initial stage first; checkpoint missing.")

    import os

    os.environ["YOLO_CONFIG_DIR"] = str(args.output.parent / "plate-local-settings")
    os.environ["WANDB_MODE"] = "disabled"
    os.environ["OMP_NUM_THREADS"] = "1"
    import hashlib
    import json
    import random
    import time
    import xml.etree.ElementTree as ET

    import torch
    from PIL import Image
    from ultralytics import YOLO, settings

    settings.update(
        {
            "sync": False,
            "wandb": False,
            "mlflow": False,
            "clearml": False,
            "comet": False,
            "tensorboard": False,
        }
    )
    torch.set_num_threads(1)
    if args.stage in {"prepare", "all"}:
        args.output.mkdir(exist_ok=False)
        rows, seen = [], set()
        for f in sorted((args.data / "annotations").glob("*.xml")):
            root = ET.parse(f).getroot()
            filename = root.findtext("filename")
            if filename is None or Path(filename).name != filename:
                continue
            image = args.data / "Pics" / filename
            if not image.is_file():
                continue
            im = Image.open(image).convert("RGB")
            w, h = im.size
            digest = hashlib.sha256(im.tobytes()).hexdigest()
            if digest in seen:
                continue
            seen.add(digest)
            boxes = []
            for obj in root.findall("object"):
                if obj.findtext("name") != "1":
                    continue
                b = obj.find("bndbox")
                if b is None:
                    continue
                x1, y1, x2, y2 = [float(b.findtext(k)) for k in ["xmin", "ymin", "xmax", "ymax"]]
                if 0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h:
                    boxes.append([x1 / w, y1 / h, x2 / w, y2 / h])
            if boxes:
                rows.append(dict(image=filename, sha256=digest, boxes=boxes))
        random.Random(42).shuffle(rows)
        rows = rows[:96]
        if len(rows) != 96:
            raise ValueError("Expected96 selected annotated images; check source version.")
        for i, row in enumerate(rows):
            row["split"] = "val" if i < 16 else "train"
            for size in [320, 640]:
                base = args.output / f"data{size}"
                for sub in ["images", "labels"]:
                    (base / sub / row["split"]).mkdir(parents=True, exist_ok=True)
                im = Image.open(args.data / "Pics" / row["image"]).convert("RGB")
                im.thumbnail((size, size))
                target = base / "images" / row["split"] / f"{i:03}.jpg"
                if size == 320:
                    im.save(target)
                else:
                    im.save(target, quality=95)
                lines = [
                    f"0 {(x1 + x2) / 2} {(y1 + y2) / 2} {x2 - x1} {y2 - y1}"
                    for x1, y1, x2, y2 in row["boxes"]
                ]
                (base / "labels" / row["split"] / f"{i:03}.txt").write_text("\n".join(lines) + "\n")
        for size in [320, 640]:
            base = (args.output / f"data{size}").resolve()
            (base / "data.yaml").write_text(
                f"path: {base}\ntrain: images/train\nval: images/val\nnames:\n  0: License Plate\n"
            )
        (args.output / "manifest.json").write_text(json.dumps(rows, indent=2))
        if args.stage == "prepare":
            return
    shared = dict(
        device="cpu",
        workers=0,
        project=str(args.output.resolve()),
        exist_ok=False,
        plots=False,
        verbose=False,
        seed=42,
        deterministic=True,
        amp=False,
        cache=False,
    )
    start = time.time()
    if args.stage in {"initial", "all"}:
        model = YOLO(str(args.weights.resolve()))
        first = model.train(
            data=str(args.output / "data320/data.yaml"),
            epochs=15,
            imgsz=320,
            batch=16,
            name="stage1",
            pretrained=True,
            **shared,
        )
        (args.output / "stage1-metrics.json").write_text(json.dumps(first.results_dict))
        if args.stage == "initial":
            return
    model = YOLO(str(args.output / "stage1/weights/best.pt"))
    final = model.train(
        data=str(args.output / "data640/data.yaml"),
        epochs=4,
        imgsz=640,
        batch=1,
        freeze=10,
        close_mosaic=0,
        name="stage2",
        **shared,
    )
    result = dict(
        source="https://www.kaggle.com/datasets/mahmoudkhater99/car-plate",
        source_version=13,
        stage1=json.loads((args.output / "stage1-metrics.json").read_text()),
        stage2=final.results_dict,
        seconds=time.time() - start,
        limits="Development only80train16dev from related captures, same reused "
        "dev selection. Not scene-independent or text OCR. Source DbCL/ODbL "
        "and Ultralytics terms separate. No raw plates or weights redistribution.",
    )
    (args.output / "report.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
