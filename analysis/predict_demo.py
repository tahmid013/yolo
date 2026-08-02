"""Generate annotated prediction images at two confidence thresholds.

For each source image, produces two side-by-side outputs so a thesis slide
can show 'model at confidence >= 0.2' next to 'model at confidence >= 0.8'.
Useful for explaining the precision/recall trade-off visually.

Usage:
    python analysis/predict_demo.py \\
        --weights runs/yolo11s_reference/weights/best.pt \\
        --images 02f4c440-139.jpg 036000eb-181.jpg \\
        --out demo/predictions

Outputs:
    demo/predictions/<stem>_conf020.jpg
    demo/predictions/<stem>_conf080.jpg
    demo/predictions/<stem>_sidebyside.jpg
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO


THRESHOLDS = [0.20, 0.80]


def annotate(image_path: Path, model: YOLO, conf: float, out_path: Path) -> dict:
    """Predict on one image at a given confidence and save annotated version."""
    results = model.predict(source=str(image_path), conf=conf, verbose=False)
    r = results[0]
    # ultralytics' plot() returns a numpy array (BGR); convert to PIL RGB
    annotated = Image.fromarray(r.plot()[..., ::-1])
    # Add a header banner with the threshold + prediction count
    banner_h = 40
    W, H = annotated.size
    canvas = Image.new("RGB", (W, H + banner_h), "white")
    canvas.paste(annotated, (0, banner_h))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("arial.ttf", 22)
    except OSError:
        font = ImageFont.load_default()
    n = len(r.boxes) if r.boxes is not None else 0
    draw.text((10, 8), f"Threshold: conf >= {conf:.2f}   |   {n} detections",
              fill="black", font=font)
    canvas.save(out_path)
    return {"threshold": conf, "n_detections": n,
            "classes": sorted({model.names[int(b.cls)] for b in r.boxes} if r.boxes is not None else set())}


def side_by_side(low_img: Path, high_img: Path, out_path: Path) -> None:
    a = Image.open(low_img)
    b = Image.open(high_img)
    W = max(a.width, b.width)
    gap = 20
    canvas = Image.new("RGB", (W * 2 + gap, max(a.height, b.height)), "white")
    canvas.paste(a, (0, 0))
    canvas.paste(b, (W + gap, 0))
    canvas.save(out_path)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", type=Path,
                    default=Path("runs/yolo11s_reference/weights/best.pt"))
    ap.add_argument("--images", type=str, nargs="+", required=True,
                    help="Image filenames (looked up under --images-dir)")
    ap.add_argument("--images-dir", type=Path,
                    default=Path("report_generation/data/images"))
    ap.add_argument("--out", type=Path, default=Path("demo/predictions"))
    args = ap.parse_args()

    if not args.weights.exists():
        print(f"ERROR: weights not found at {args.weights}", file=sys.stderr)
        return 1

    args.out.mkdir(parents=True, exist_ok=True)
    print(f"[demo] loading {args.weights}")
    model = YOLO(str(args.weights))

    for img_name in args.images:
        img_path = args.images_dir / img_name
        if not img_path.exists():
            print(f"[demo] SKIP {img_name}: not found in {args.images_dir}")
            continue
        stem = img_path.stem
        print(f"\n[demo] {img_name}")
        out_low = args.out / f"{stem}_conf020.jpg"
        out_high = args.out / f"{stem}_conf080.jpg"
        info_low = annotate(img_path, model, THRESHOLDS[0], out_low)
        info_high = annotate(img_path, model, THRESHOLDS[1], out_high)
        print(f"  conf>=0.20 -> {info_low['n_detections']} detections "
              f"[classes: {', '.join(info_low['classes'])}]  ->  {out_low.name}")
        print(f"  conf>=0.80 -> {info_high['n_detections']} detections "
              f"[classes: {', '.join(info_high['classes'])}]  ->  {out_high.name}")
        side = args.out / f"{stem}_sidebyside.jpg"
        side_by_side(out_low, out_high, side)
        print(f"  side-by-side saved: {side.name}")

    print(f"\n[demo] outputs in {args.out.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
