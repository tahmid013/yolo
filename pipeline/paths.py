"""Standard paths used by the Colab-side pipeline.

Override DRIVE_ROOT if you mounted Drive at a non-default location.
"""
from __future__ import annotations

from pathlib import Path

DRIVE_ROOT = Path("/content/drive/MyDrive/yolo-pipeline")
DATASETS_DIR = DRIVE_ROOT / "datasets"
RUNS_DIR = DRIVE_ROOT / "runs"

LOCAL_DATASET = Path("/content/dataset")
LOCAL_RUNS = Path("/content/runs")


def dataset_zip(version: str) -> Path:
    return DATASETS_DIR / version / "dataset.zip"


def dataset_meta(version: str) -> Path:
    return DATASETS_DIR / version / "dataset.meta.json"
