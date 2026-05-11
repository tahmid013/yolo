"""Materialize the dataset on Colab (unzip from Drive, verify SHA256)."""
from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path


def _sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def ensure_dataset(
    zip_path: str | Path,
    meta_path: str | Path,
    target: str | Path = "/content/dataset",
    *,
    force: bool = False,
) -> Path:
    """Unzip dataset to `target`, verifying SHA256 against meta_path.

    Idempotent: if `target/data.yaml` already exists and `force=False`, skip unzip.
    """
    zip_path = Path(zip_path)
    meta_path = Path(meta_path)
    target = Path(target)

    if not zip_path.exists():
        raise FileNotFoundError(f"Dataset zip not found at {zip_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"Dataset meta not found at {meta_path}")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    expected_sha = meta.get("zip", {}).get("sha256")
    if not expected_sha:
        raise ValueError(f"meta {meta_path} has no zip.sha256")

    print(f"[dataset] verifying sha256 of {zip_path.name} ...")
    actual_sha = _sha256_file(zip_path)
    if actual_sha != expected_sha:
        raise ValueError(
            f"SHA256 mismatch for {zip_path.name}:\n"
            f"  expected: {expected_sha}\n"
            f"  actual:   {actual_sha}"
        )
    print(f"[dataset] sha256 OK ({expected_sha[:16]}...)")

    data_yaml = target / "data.yaml"
    if data_yaml.exists() and not force:
        print(f"[dataset] {data_yaml} already exists, skipping unzip")
        return data_yaml

    target.parent.mkdir(parents=True, exist_ok=True)
    print(f"[dataset] unzipping to {target.parent} ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(target.parent)

    if not data_yaml.exists():
        raise RuntimeError(
            f"After unzip, expected {data_yaml} to exist. "
            f"Check that the zip has a `dataset/` prefix."
        )

    counts = meta.get("counts", {})
    print(
        f"[dataset] ready: train={counts.get('train','?')} "
        f"val={counts.get('val','?')} test={counts.get('test','?')}"
    )
    return data_yaml
