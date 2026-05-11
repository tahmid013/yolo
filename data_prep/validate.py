"""Discover and validate image/label pairs in a YOLO-format dataset directory."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".webp"}


@dataclass
class Pair:
    image: Path
    label: Path
    stem: str


@dataclass
class ValidationReport:
    pairs: list[Pair]
    skipped_orphan_images: list[str]
    skipped_orphan_labels: list[str]
    label_errors: dict[str, list[str]]


def discover_pairs(src: Path) -> tuple[list[Pair], list[str], list[str]]:
    images_dir = src / "images"
    labels_dir = src / "labels"
    if not images_dir.is_dir() or not labels_dir.is_dir():
        raise FileNotFoundError(
            f"Expected {images_dir} and {labels_dir} to exist"
        )

    image_by_stem: dict[str, Path] = {}
    for p in images_dir.iterdir():
        if p.suffix.lower() in IMAGE_EXTS and p.is_file():
            image_by_stem[p.stem] = p

    label_by_stem: dict[str, Path] = {
        p.stem: p for p in labels_dir.glob("*.txt") if p.is_file()
    }

    pairs: list[Pair] = []
    orphan_images: list[str] = []
    orphan_labels: list[str] = []

    for stem, img in image_by_stem.items():
        lbl = label_by_stem.get(stem)
        if lbl is None:
            orphan_images.append(img.name)
        else:
            pairs.append(Pair(image=img, label=lbl, stem=stem))

    for stem, lbl in label_by_stem.items():
        if stem not in image_by_stem:
            orphan_labels.append(lbl.name)

    pairs.sort(key=lambda p: p.stem)
    return pairs, orphan_images, orphan_labels


def validate_label_file(path: Path, num_classes: int) -> list[str]:
    """Return a list of error messages for this label file (empty = valid)."""
    errors: list[str] = []
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as e:
        return [f"unreadable: {e}"]

    for i, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            errors.append(f"line {i}: expected 5 fields, got {len(parts)}")
            continue
        try:
            cls = int(parts[0])
            x, y, w, h = (float(v) for v in parts[1:])
        except ValueError as e:
            errors.append(f"line {i}: non-numeric ({e})")
            continue
        if not (0 <= cls < num_classes):
            errors.append(f"line {i}: class {cls} out of range [0,{num_classes})")
        for name, v in (("x", x), ("y", y), ("w", w), ("h", h)):
            if not (0.0 <= v <= 1.0):
                errors.append(f"line {i}: {name}={v} not in [0,1]")
    return errors


def validate_pairs(src: Path, num_classes: int) -> ValidationReport:
    pairs, orphan_images, orphan_labels = discover_pairs(src)
    label_errors: dict[str, list[str]] = {}
    valid_pairs: list[Pair] = []
    for p in pairs:
        errs = validate_label_file(p.label, num_classes)
        if errs:
            label_errors[p.label.name] = errs
        else:
            valid_pairs.append(p)
    return ValidationReport(
        pairs=valid_pairs,
        skipped_orphan_images=orphan_images,
        skipped_orphan_labels=orphan_labels,
        label_errors=label_errors,
    )
