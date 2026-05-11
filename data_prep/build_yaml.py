"""Emit a Ultralytics-compatible data.yaml with Colab-relative paths."""
from __future__ import annotations

from pathlib import Path

import yaml


def write_data_yaml(
    out: Path,
    names: list[str],
    colab_root: str = "/content/dataset",
) -> Path:
    payload = {
        "path": colab_root,
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "nc": len(names),
        "names": names,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False, allow_unicode=True)
    return out


def read_class_names(classes_txt: Path) -> list[str]:
    """Read class names from a one-per-line file. Skips blank lines.

    Order is preserved; class index 0 is the first non-blank line.
    """
    names = [
        line.strip()
        for line in classes_txt.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not names:
        raise ValueError(f"{classes_txt} contains no class names")
    return names
