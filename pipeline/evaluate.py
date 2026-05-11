"""Evaluate a trained run on the held-out test split and persist results.

Ultralytics' `model.val(...)` defaults to split="val". We always pass split="test"
explicitly so the analysis report has true held-out numbers.
"""
from __future__ import annotations

import json
from pathlib import Path

from . import meta as meta_mod
from . import paths as paths_mod
from . import persist as persist_mod


def run(
    run_dir: str | Path,
    *,
    data_yaml: str | Path = paths_mod.LOCAL_DATASET / "data.yaml",
    split: str = "test",
    drive_runs_dir: str | Path = paths_mod.RUNS_DIR,
    push_to_drive: bool = True,
) -> dict:
    from ultralytics import YOLO

    run_dir = Path(run_dir)
    data_yaml = Path(data_yaml)
    drive_runs_dir = Path(drive_runs_dir)

    weights = run_dir / "weights" / "best.pt"
    if not weights.exists():
        raise FileNotFoundError(f"weights/best.pt not found under {run_dir}")
    if not data_yaml.exists():
        raise FileNotFoundError(f"data.yaml not found at {data_yaml}")

    print(f"[evaluate] {run_dir.name} on split={split}")
    y = YOLO(str(weights))
    metrics = y.val(
        data=str(data_yaml),
        split=split,
        project=str(run_dir),
        name="eval",
        exist_ok=True,
        plots=True,
        save_json=True,
    )

    payload = _build_payload(metrics, data_yaml, split)
    eval_dir = run_dir / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    (eval_dir / "per_class.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    meta_mod.update_run_meta(run_dir, {"test": payload})
    print(f"[evaluate] wrote {eval_dir / 'per_class.json'}")

    if push_to_drive:
        persist_mod.copy_to_drive(run_dir, drive_runs_dir)

    return payload


def _class_names(data_yaml: Path) -> list[str]:
    import yaml

    cfg = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    return list(cfg.get("names", []))


def _build_payload(metrics, data_yaml: Path, split: str) -> dict:
    names = _class_names(data_yaml)
    box = getattr(metrics, "box", None)

    overall = {
        "mAP50": _safe_float(getattr(box, "map50", None)) if box else None,
        "mAP50_95": _safe_float(getattr(box, "map", None)) if box else None,
        "precision": _safe_float(_mean(getattr(box, "mp", None))) if box else None,
        "recall": _safe_float(_mean(getattr(box, "mr", None))) if box else None,
    }

    per_class: list[dict] = []
    if box is not None:
        try:
            # Do NOT use `arr or []` — on a numpy array that triggers
            # "truth value of an array with more than one element is ambiguous".
            maps = _as_list(getattr(box, "maps", None))           # per-class mAP50-95, len=nc
            ap_class_index = _as_list(getattr(box, "ap_class_index", None))
            ap50 = _as_list(getattr(box, "ap50", None))           # per detected class
            p = _as_list(getattr(box, "p", None))                 # per detected class
            r = _as_list(getattr(box, "r", None))                 # per detected class
            for i, cls_idx in enumerate(ap_class_index):
                cls_idx = int(cls_idx)
                per_class.append({
                    "id": cls_idx,
                    "name": names[cls_idx] if 0 <= cls_idx < len(names) else str(cls_idx),
                    "ap50":      _safe_float(ap50[i]) if i < len(ap50) else None,
                    "ap50_95":   _safe_float(maps[cls_idx]) if cls_idx < len(maps) else None,
                    "precision": _safe_float(p[i]) if i < len(p) else None,
                    "recall":    _safe_float(r[i]) if i < len(r) else None,
                })
        except Exception as e:
            print(f"[evaluate] WARN: could not build per-class metrics: {e}")

    return {"split": split, "overall": overall, "per_class": per_class}


def _as_list(v) -> list:
    """Convert a numpy array / sequence / None into a plain Python list.

    Avoids `arr or []` patterns that fail on numpy arrays with the
    'truth value is ambiguous' error.
    """
    if v is None:
        return []
    try:
        return list(v)
    except TypeError:
        return []


def _safe_float(v) -> float | None:
    try:
        if v is None:
            return None
        return float(v)
    except (TypeError, ValueError):
        return None


def _mean(v) -> float | None:
    if v is None:
        return None
    try:
        seq = list(v)
        if not seq:
            return None
        return sum(seq) / len(seq)
    except TypeError:
        try:
            return float(v)
        except (TypeError, ValueError):
            return None
