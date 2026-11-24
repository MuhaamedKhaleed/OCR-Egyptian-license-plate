"""Text-controlled Egyptian-style synthetic plate experiment, explicit local run only."""

from pathlib import Path
import argparse
import json
import random
import hashlib

LETTERS = list("أبجدرسصطعفقلمنهوي")
DIGITS = list("١٢٣٤٥٦٧٨٩")
FONTS = [
    "/usr/share/fonts/truetype/noto/NotoSansArabic-Bold.ttf",
    "/usr/share/fonts/truetype/noto/NotoNaskhArabic-Bold.ttf",
    "/usr/share/fonts/truetype/noto/NotoKufiArabic-Bold.ttf",
]


def generate(out, fonts=None, latin_font=None):
    fonts = fonts or FONTS
    latin_font = latin_font or "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    for font_path in [*fonts, latin_font]:
        if not Path(font_path).is_file():
            raise ValueError(
                "Font missing. Pass --font and --latin-font with installed local files."
            )
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont, ImageFilter

    out.mkdir(exist_ok=False, parents=True)
    rng = random.Random(20261125)
    rows = []
    seen = set()
    for i in range(500):
        while True:
            digits = "".join(rng.choices(DIGITS, k=3))
            letters = "".join(rng.choices(LETTERS, k=3))
            if (digits, letters) not in seen:
                seen.add((digits, letters))
                break
        split = "train" if i < 400 else "dev" if i < 450 else "test"
        im = Image.new("RGB", (350, 170), (rng.randint(210, 250),) * 3)
        d = ImageDraw.Draw(im)
        d.rounded_rectangle((2, 2, 347, 167), radius=8, outline=(35, 35, 35), width=3)
        d.rectangle(
            (5, 5, 344, 57),
            fill=(rng.randint(70, 110), rng.randint(140, 185), rng.randint(185, 230)),
        )
        font = ImageFont.truetype(fonts[0], 27)
        d.text((20, 11), "EGYPT", font=ImageFont.truetype(latin_font, 23), fill="black")
        d.text((255, 2), "مصر", font=font, fill="black")
        d.line((175, 59, 175, 161), fill=(70, 70, 70), width=2)
        chars = list(digits) + list(reversed(letters))
        boxes = []
        chosen_font = rng.choice(fonts)
        for j, c in enumerate(chars):
            x = 20 + j * 49 if j < 3 else 194 + (j - 3) * 49
            box = [x, 66, x + 43, 151]
            boxes.append(box)
            glyph = "هـ" if c == "ه" else "ى" if c == "ي" else c
            f = ImageFont.truetype(chosen_font, rng.randint(48, 62))
            b = d.textbbox((0, 0), glyph, font=f)
            layer = Image.new("RGBA", (b[2] - b[0] + 4, b[3] - b[1] + 4), (0, 0, 0, 0))
            ImageDraw.Draw(layer).text(
                (2 - b[0], 2 - b[1]), glyph, font=f, fill=(rng.randint(0, 35),) * 3 + (255,)
            )
            scale = min(36 / layer.width, 66 / layer.height, 1.0)
            layer = layer.resize(
                (max(1, int(layer.width * scale)), max(1, int(layer.height * scale))),
                Image.Resampling.LANCZOS,
            )
            im.paste(layer, (x + (43 - layer.width) // 2, 106 - layer.height // 2), layer)
        d = ImageDraw.Draw(im)
        for x in [11, 339]:
            d.ellipse((x - 3, 29, x + 3, 35), fill=(120, 120, 120))
        # Blur, contrast, lighting and sensor noise. No claim this captures real domain.
        if rng.random() < 0.65:
            im = im.filter(ImageFilter.GaussianBlur(rng.uniform(0.1, 0.85)))
        a = np.asarray(im).astype(np.float32)
        nr = np.random.default_rng(7000 + i)
        a = a * rng.uniform(0.80, 1.08) + nr.normal(0, rng.uniform(1, 6), a.shape)
        im = Image.fromarray(a.clip(0, 255).astype("uint8"))
        p = out / split / f"{i:04}.jpg"
        p.parent.mkdir(exist_ok=True)
        im.save(p, quality=92, subsampling=0)
        rows.append(
            dict(
                file=str(p.relative_to(out)),
                split=split,
                digits=digits,
                letters_rtl=letters,
                visual_left_to_right="".join(chars),
                boxes=boxes,
                font_scope="Noto Arabic OFL1.1",
            )
        )
    (out / "manifest.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2))
    canvas = Image.new("RGB", (1050, 850), "#ddd")
    for k, i in enumerate([0, 1, 2, 3, 4, 5, 100, 150, 200, 250, 300, 350, 400, 450, 499]):
        canvas.paste(Image.open(out / rows[i]["file"]), (k % 3 * 350, k // 3 * 170))
    canvas.save(out / "contact-sheet.png")
    print(
        json.dumps(
            {
                "images": len(rows),
                "train": 400,
                "dev": 50,
                "test": 50,
                "unique_registrations": len(seen),
                "alphabet": LETTERS,
                "digits": DIGITS,
            },
            ensure_ascii=False,
        )
    )


def validate_data(data):
    rows = json.loads((data / "manifest.json").read_text(encoding="utf-8"))
    if not isinstance(rows, list) or len(rows) != 500:
        raise ValueError("Expected500 manifest rows.")
    counts = {k: 0 for k in ["train", "dev", "test"]}
    files = set()
    registrations = set()
    hashes = set()
    for r in rows:
        if not isinstance(r, dict):
            raise ValueError("Invalid manifest row.")
        rel = Path(r["file"])
        if rel.is_absolute() or ".." in rel.parts:
            raise ValueError("Unsafe image path.")
        p = data / rel
        if not p.resolve().is_relative_to(data.resolve()):
            raise ValueError("Image leaves dataset directory.")
        if r["split"] not in counts:
            raise ValueError("Invalid split.")
        counts[r["split"]] += 1
        if len(r["digits"]) != 3 or any(c not in DIGITS for c in r["digits"]):
            raise ValueError("Invalid digits.")
        if len(r["letters_rtl"]) != 3 or any(c not in LETTERS for c in r["letters_rtl"]):
            raise ValueError("Invalid letters.")
        chars = r["digits"] + r["letters_rtl"][::-1]
        if r["visual_left_to_right"] != chars:
            raise ValueError("Text order mismatch.")
        registration = (r["digits"], r["letters_rtl"])
        if r["file"] in files or registration in registrations:
            raise ValueError("Duplicate file or registration.")
        files.add(r["file"])
        registrations.add(registration)
        digest = hashlib.sha256(p.read_bytes()).hexdigest()
        if digest in hashes:
            raise ValueError("Identical image across manifest rows.")
        hashes.add(digest)
        from PIL import Image

        with Image.open(p) as im:
            w, h = im.size
            if len(r["boxes"]) != 6:
                raise ValueError("Expected six character boxes.")
            for b in r["boxes"]:
                if len(b) != 4 or any(type(x) is not int for x in b):
                    raise ValueError("Box coordinates must be integers.")
                x1, y1, x2, y2 = b
                if not 0 <= x1 < x2 <= w or not 0 <= y1 < y2 <= h:
                    raise ValueError("Character box outside image.")
            im.verify()
    if counts != {"train": 400, "dev": 50, "test": 50}:
        raise ValueError("Expected400/50/50split.")
    return rows


def train(data, out):
    import numpy as np
    from PIL import Image
    import torch
    from torch import nn

    rows = validate_data(data)
    torch.set_num_threads(1)
    torch.manual_seed(42)
    out.mkdir(parents=True, exist_ok=False)
    alphabet = DIGITS + LETTERS

    def prep(split):
        a = []
        y = []
        for r in rows:
            if r["split"] != split:
                continue
            im = Image.open(data / r["file"]).convert("L")
            for c, b in zip(r["visual_left_to_right"], r["boxes"], strict=True):
                a.append(np.asarray(im.crop(b).resize((32, 48)), np.float32)[None] / 255)
                y.append(alphabet.index(c))
        return torch.tensor(np.stack(a)), torch.tensor(y)

    x, y = prep("train")
    dv, dy = prep("dev")
    model = nn.Sequential(
        nn.Conv2d(1, 16, 3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(16, 32, 3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(32 * 12 * 8, 128),
        nn.ReLU(),
        nn.Linear(128, len(alphabet)),
    )
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    loss = nn.CrossEntropyLoss()
    best = -1
    history = []
    for ep in range(1, 26):
        model.train()
        idx = torch.randperm(len(x))
        for b in idx.split(64):
            opt.zero_grad()
            logits = model(x[b])
            v = loss(logits, y[b])
            v.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            acc = float((model(dv).argmax(1) == dy).float().mean())
        history.append({"epoch": ep, "dev_char_accuracy": acc})
        if acc > best:
            best = acc
            torch.save(model.state_dict(), out / "best.pt")
        print(history[-1], flush=True)
    model.load_state_dict(torch.load(out / "best.pt", weights_only=True))
    model.eval()
    te, ty = prep("test")
    with torch.no_grad():
        pred = model(te).argmax(1)
    ok = (pred == ty).numpy().reshape(-1, 6)
    report = {
        "training_plates": 400,
        "dev_plates": 50,
        "test_plates": 50,
        "training_glyphs": len(x),
        "alphabet": alphabet,
        "history": history,
        "selected_dev_char_accuracy": best,
        "test_char_accuracy": float(ok.mean()),
        "test_plate_exact_match": float(ok.all(1).mean()),
        "limits": [
            "All images synthetic, Cairo-style 3letters3digits subset, not all Egyptian formats.",
            "Known renderer character boxes used for cropping, oracle segmentation. Not end-to-end OCR or real-image accuracy.",
            "Unique registration strings across splits, shared font/layout families, not out-of-domain evaluation.",
            "Random fictional strings may coincide with real registrations by chance; no ownership or registration lookup.",
            "No security seals or authentic plate production assets reproduced.",
        ],
    }
    (out / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps({k: v for k, v in report.items() if k != "history"}, ensure_ascii=False))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=["generate", "train"])
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path)
    p.add_argument("--font", action="append")
    p.add_argument("--latin-font")
    a = p.parse_args()
    if a.stage == "generate":
        generate(a.data, a.font, a.latin_font)
    else:
        if a.output is None:
            p.error("--output is required for train")
        train(a.data, a.output)
