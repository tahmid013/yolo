"""Build a Colab-ready dataset zip from the reference YOLO dataset.

Usage:
    python data_prep/prepare_dataset.py \
        --src report_generation/data \
        --out data_prep/build \
        --seed 42 \
        --version v1

Output:
    <out>/dataset/{train,val,test}/{images,labels}/...
    <out>/splits/{train,val,test}.txt          one stem per line (audit trail)
    <out>/data.yaml                            Ultralytics config
    <out>/dataset.zip                          upload to Drive
    <out>/dataset.meta.json                    counts, sha256, env, etc.
    <out>/skipped.json                         orphans + label errors
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import platform
import shutil
import sys
import zipfile
from collections import Counter
from pathlib import Path

from PIL import Image

# Allow running as a script (python data_prep/prepare_dataset.py)
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from data_prep.build_yaml import read_class_names, write_data_yaml
    from data_prep.split import SplitResult, split_pairs
    from data_prep.validate import Pair, validate_pairs
else:
    from .build_yaml import read_class_names, write_data_yaml
    from .split import SplitResult, split_pairs
    from .validate import Pair, validate_pairs


CONVERT_EXTS = {".webp", ".jpeg"}


def _materialize_pair(
    pair: Pair,
    split_dir: Path,
    convert: bool,
) -> bool:
    """Copy (or convert) one image+label pair into split_dir/{images,labels}/.

    Returns True if the image was converted to .jpg.
    """
    images_out = split_dir / "images"
    labels_out = split_dir / "labels"
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    src_ext = pair.image.suffix.lower()
    converted = False
    if convert and src_ext in CONVERT_EXTS:
        dst_image = images_out / f"{pair.stem}.jpg"
        with Image.open(pair.image) as im:
            im.convert("RGB").save(dst_image, format="JPEG", quality=95)
        converted = True
    else:
        dst_image = images_out / pair.image.name
        shutil.copy2(pair.image, dst_image)

    shutil.copy2(pair.label, labels_out / pair.label.name)
    return converted


def _class_histogram(pairs: list[Pair], num_classes: int) -> dict[int, int]:
    counter: Counter[int] = Counter()
    for p in pairs:
        for line in p.label.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                cls = int(line.split()[0])
            except (ValueError, IndexError):
                continue
            if 0 <= cls < num_classes:
                counter[cls] += 1
    return {i: counter.get(i, 0) for i in range(num_classes)}


def _sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _zip_dataset(dataset_root: Path, out_zip: Path) -> None:
    """Zip dataset_root into out_zip with arcnames prefixed by `dataset/`.

    Unzipping to /content/ yields /content/dataset/{train,val,test,data.yaml}.
    """
    out_zip.parent.mkdir(parents=True, exist_ok=True)
    if out_zip.exists():
        out_zip.unlink()
    parent = dataset_root.parent
    with zipfile.ZipFile(
        out_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
    ) as zf:
        for p in sorted(dataset_root.rglob("*")):
            if p.is_file():
                zf.write(p, arcname=p.relative_to(parent))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", type=Path, required=True,
                    help="Source dataset dir (contains images/ and labels/)")
    ap.add_argument("--out", type=Path, required=True,
                    help="Output build dir (will be wiped and recreated)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--split", type=float, nargs=3, default=(0.8, 0.1, 0.1),
                    metavar=("TRAIN", "VAL", "TEST"))
    ap.add_argument("--version", type=str, default="v1")
    ap.add_argument("--classes", type=Path, default=None,
                    help="Path to classes.txt (defaults to <src>/classes.txt)")
    ap.add_argument("--keep-ext", action="store_true",
                    help="Do NOT convert .webp/.jpeg to .jpg")
    args = ap.parse_args()

    src: Path = args.src.resolve()
    out: Path = args.out.resolve()
    classes_txt = (args.classes or (src / "classes.txt")).resolve()

    names = read_class_names(classes_txt)
    num_classes = len(names)
    print(f"[1/8] Loaded {num_classes} class names from {classes_txt}")

    print(f"[2/8] Validating pairs under {src} ...")
    report = validate_pairs(src, num_classes)
    print(f"      valid pairs: {len(report.pairs)}")
    print(f"      orphan images (no label): {len(report.skipped_orphan_images)}")
    print(f"      orphan labels (no image): {len(report.skipped_orphan_labels)}")
    print(f"      label files with errors:  {len(report.label_errors)}")

    if not report.pairs:
        print("ERROR: no valid pairs found", file=sys.stderr)
        return 1

    print(f"[3/8] Wiping and recreating {out} ...")
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    skipped = {
        "orphan_images": report.skipped_orphan_images,
        "orphan_labels": report.skipped_orphan_labels,
        "label_errors": report.label_errors,
    }
    (out / "skipped.json").write_text(
        json.dumps(skipped, indent=2), encoding="utf-8"
    )

    print(f"[4/8] Splitting (ratios={args.split}, seed={args.seed}) ...")
    splits: SplitResult = split_pairs(report.pairs, tuple(args.split), args.seed)
    split_map: dict[str, list[Pair]] = {
        "train": splits.train, "val": splits.val, "test": splits.test
    }
    for name, pairs in split_map.items():
        print(f"      {name}: {len(pairs)}")

    splits_dir = out / "splits"
    splits_dir.mkdir()
    for name, pairs in split_map.items():
        (splits_dir / f"{name}.txt").write_text(
            "\n".join(p.stem for p in pairs) + "\n", encoding="utf-8"
        )

    print(f"[5/8] Materializing dataset into {out}/dataset/ ...")
    dataset_root = out / "dataset"
    converted = 0
    for name, pairs in split_map.items():
        split_dir = dataset_root / name
        for p in pairs:
            if _materialize_pair(p, split_dir, convert=not args.keep_ext):
                converted += 1
    print(f"      converted to .jpg: {converted}")

    print("[6/8] Writing data.yaml ...")
    data_yaml = write_data_yaml(dataset_root / "data.yaml", names)
    # Also keep a copy at <out>/data.yaml for inspection
    shutil.copy2(data_yaml, out / "data.yaml")

    print("[7/8] Zipping dataset ...")
    out_zip = out / "dataset.zip"
    _zip_dataset(dataset_root, out_zip)
    sha = _sha256_file(out_zip)
    print(f"      {out_zip.name}: {out_zip.stat().st_size/1e6:.1f} MB  sha256={sha[:16]}...")

    print("[8/8] Writing dataset.meta.json ...")
    meta = {
        "version": args.version,
        "seed": args.seed,
        "split_ratios": list(args.split),
        "counts": {name: len(pairs) for name, pairs in split_map.items()},
        "class_names": names,
        "class_distribution": {
            name: _class_histogram(pairs, num_classes)
            for name, pairs in split_map.items()
        },
        "skipped": {
            "orphan_images": len(report.skipped_orphan_images),
            "orphan_labels": len(report.skipped_orphan_labels),
            "label_errors": len(report.label_errors),
        },
        "converted_to_jpg": converted,
        "source_root": str(src),
        "zip": {
            "name": out_zip.name,
            "size_bytes": out_zip.stat().st_size,
            "sha256": sha,
        },
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "tool_versions": {
            "python": platform.python_version(),
            "pillow": Image.__version__,
        },
    }
    (out / "dataset.meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )

    print("")
    print("Done.")
    print(f"  Upload these to MyDrive/yolo-pipeline/datasets/{args.version}/:")
    print(f"    {out_zip}")
    print(f"    {out / 'dataset.meta.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
