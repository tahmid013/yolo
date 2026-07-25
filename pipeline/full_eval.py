"""Re-evaluate a trained best.pt on the FULL reference dataset (all 1,026 images).

Matches the methodology used in report_generation/yolo_research_final_v1.docx:
the reference team ran `model.val(data=data.yaml)` on the entire 1,026-image
set (no train/val/test split). Numbers include training-set leakage — this is
intentional so v11 and v12 tables are apples-to-apples in the final docx.

Output: <run_dir>/full_eval/per_class.json with Precision / Recall / mAP@0.5 /
mAP@0.5:0.95 per class, plus `full_eval` field added to run_meta.json.
"""
from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

import yaml

from . import meta as meta_mod
from . import persist as persist_mod


def _make_full_data_yaml(source_dataset_dir: Path, tmp_root: Path) -> Path:
    """Build a data.yaml where train/val/test all point at the same full folder.

    Ultralytics requires images/ + labels/ subfolders. The reference dataset
    layout at report_generation/data has `images/` and `labels/` — we symlink
    (or copy) into a small scaffold where train/val/test all resolve to the
    same full set.
    """
    root = tmp_root / "full_dataset"
    root.mkdir(parents=True, exist_ok=True)
    # Point 'all' split at the source
    all_dir = root / "all"
    all_dir.mkdir(exist_ok=True)
    for sub in ("images", "labels"):
        target = all_dir / sub
        if target.exists():
            if target.is_symlink() or target.is_file():
                target.unlink()
            else:
                shutil.rmtree(target)
        # Try symlink first (fast, no copy); fall back to copytree if permission denied
        try:
            target.symlink_to((source_dataset_dir / sub).resolve(),
                              target_is_directory=True)
        except (OSError, NotImplementedError):
            shutil.copytree(source_dataset_dir / sub, target)

    # class names from classes.txt if available, else from run_meta
    classes_txt = source_dataset_dir / "classes.txt"
    if classes_txt.exists():
        names = [ln.strip() for ln in classes_txt.read_text(encoding="utf-8").splitlines()
                 if ln.strip()]
    else:
        raise FileNotFoundError(f"classes.txt not found at {classes_txt}")

    yaml_path = root / "data.yaml"
    yaml_path.write_text(yaml.safe_dump({
        "path": str(root),
        "train": "all/images",
        "val":   "all/images",
        "test":  "all/images",
        "nc": len(names),
        "names": names,
    }, sort_keys=False), encoding="utf-8")
    return yaml_path


def run(
    run_dir: str | Path,
    source_dataset_dir: str | Path,
    *,
    drive_runs_dir: str | Path | None = None,
    push_to_drive: bool = True,
) -> dict:
    """Run YOLO.val on ALL 1,026 images and save results."""
    from ultralytics import YOLO

    run_dir = Path(run_dir)
    source_dataset_dir = Path(source_dataset_dir)

    weights = run_dir / "weights" / "best.pt"
    if not weights.exists():
        raise FileNotFoundError(f"weights/best.pt not found under {run_dir}")

    tmp_root = Path(tempfile.mkdtemp(prefix="fulleval_"))
    try:
        data_yaml = _make_full_data_yaml(source_dataset_dir, tmp_root)
        print(f"[full_eval] {run_dir.name}: evaluating on full set at {data_yaml}")

        y = YOLO(str(weights))
        # split='val' is fine — data.yaml has val pointing at the full set
        metrics = y.val(
            data=str(data_yaml),
            split="val",
            project=str(run_dir),
            name="full_eval",
            exist_ok=True,
            plots=True,
            save_json=True,
        )

        names = _class_names(data_yaml)
        payload = _build_payload(metrics, names)
        (run_dir / "full_eval" / "per_class.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )
        meta_mod.update_run_meta(run_dir, {"full_eval": payload})
        print(f"[full_eval] {run_dir.name}: overall mAP50={payload['overall']['mAP50']:.4f}")

        if push_to_drive and drive_runs_dir is not None:
            persist_mod.copy_to_drive(run_dir, Path(drive_runs_dir))

        return payload
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


def _class_names(data_yaml: Path) -> list[str]:
    cfg = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    return list(cfg.get("names", []))


def _as_list(v) -> list:
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


def _build_payload(metrics, names: list[str]) -> dict:
    box = getattr(metrics, "box", None)
    overall = {
        "mAP50":     _safe_float(getattr(box, "map50", None)) if box else None,
        "mAP50_95":  _safe_float(getattr(box, "map", None)) if box else None,
        "precision": _safe_float(getattr(box, "mp", None)) if box else None,
        "recall":    _safe_float(getattr(box, "mr", None)) if box else None,
    }

    per_class: list[dict] = []
    if box is not None:
        try:
            maps = _as_list(getattr(box, "maps", None))
            ap_class_index = _as_list(getattr(box, "ap_class_index", None))
            ap50 = _as_list(getattr(box, "ap50", None))
            p = _as_list(getattr(box, "p", None))
            r = _as_list(getattr(box, "r", None))
            # nt_per_class holds instance counts per class (0..nc-1) as tensor
            nt = _as_list(getattr(getattr(metrics, "confusion_matrix", None),
                                  "nc", None))  # fallback below
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
            print(f"[full_eval] WARN: could not build per-class metrics: {e}")

    return {"split": "full", "overall": overall, "per_class": per_class}
