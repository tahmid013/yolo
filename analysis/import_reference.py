"""Convert the reference team's YOLOv11 training folders into runs/-compatible dirs.

Copies weights + results.csv + args.yaml + plots from
`report_generation/YOLOv11{n,s,m,l}/.../my_model/train/` into
`runs/yolo11{n,s,m,l}_reference/`, and synthesizes a run_meta.json (val metrics
from the last row of results.csv) so `analysis/build_report.py` picks them up
alongside freshly-trained v12 runs.

Test-set metrics are NOT filled in — the reference team only reported val on
their own split. To fill test-split metrics, upload the produced folders to
Drive and run `pipeline.evaluate.run(run_dir=..., split='test')` on each.

Usage:
    python analysis/import_reference.py --src report_generation --out runs
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import shutil
import sys
from pathlib import Path


VARIANTS = ("yolo11n", "yolo11s", "yolo11m", "yolo11l")

# Files (relative to a reference train dir) worth carrying over. Keep the list
# tight — no train_batch*.jpg (huge) or the results.png (regenerable).
COPY_FILES = (
    "results.csv",
    "args.yaml",
    "confusion_matrix.png",
    "confusion_matrix_normalized.png",
    "BoxPR_curve.png",
    "BoxP_curve.png",
    "BoxR_curve.png",
    "BoxF1_curve.png",
)


_VARIANT_TO_DIR = {
    "yolo11n": "YOLOv11n",
    "yolo11s": "YOLOv11s",
    "yolo11m": "YOLOv11m",
    "yolo11l": "YOLOv11l",
}


def _find_train_dir(src_root: Path, variant: str) -> Path | None:
    """Walk src_root looking for a train/ folder whose parent chain names the variant.

    Reference layout is inconsistent (some nested twice), so glob for any
    train/ folder that contains results.csv under YOLOv11X/.
    """
    dir_name = _VARIANT_TO_DIR.get(variant)
    if dir_name is None:
        return None
    base = src_root / dir_name
    if not base.is_dir():
        return None
    candidates = sorted(base.rglob("results.csv"))
    if not candidates:
        return None
    # Pick the train dir (parent of results.csv) with the most complete weights folder
    def score(p: Path) -> int:
        train_dir = p.parent
        weights = train_dir / "weights"
        return (weights / "best.pt").exists() + (weights / "last.pt").exists()
    best = max(candidates, key=score)
    return best.parent


def _tail_row(csv_path: Path) -> dict | None:
    with csv_path.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    last = rows[-1]
    out: dict = {}
    for k, v in last.items():
        if k is None:
            continue
        k = k.strip()
        try:
            out[k] = float(v)
        except (TypeError, ValueError):
            out[k] = v
    return out


def _copy_run(train_dir: Path, dst: Path) -> tuple[bool, int, int]:
    """Copy the essential files. Returns (has_best, epochs, wall_time_sec_int)."""
    dst.mkdir(parents=True, exist_ok=True)
    (dst / "weights").mkdir(exist_ok=True)

    weights_src = train_dir / "weights"
    for name in ("best.pt", "last.pt"):
        s = weights_src / name
        if s.exists():
            shutil.copy2(s, dst / "weights" / name)
    has_best = (dst / "weights" / "best.pt").exists()

    for name in COPY_FILES:
        s = train_dir / name
        if s.exists():
            shutil.copy2(s, dst / name)

    # epochs completed + wall time from results.csv (last row)
    epochs = 0
    wall = 0
    csv_path = dst / "results.csv"
    if csv_path.exists():
        row = _tail_row(csv_path) or {}
        try:
            epochs = int(row.get("epoch") or 0)
        except (TypeError, ValueError):
            epochs = 0
        try:
            wall = int(float(row.get("time") or 0))
        except (TypeError, ValueError):
            wall = 0
    return has_best, epochs, wall


def _synth_run_meta(dst: Path, model: str, epochs: int, wall: int) -> None:
    """Write a run_meta.json compatible with analysis/collect.load_runs."""
    final = {}
    csv_path = dst / "results.csv"
    if csv_path.exists():
        row = _tail_row(csv_path) or {}
        for k in (
            "metrics/mAP50(B)", "metrics/mAP50-95(B)",
            "metrics/precision(B)", "metrics/recall(B)",
            "train/box_loss", "val/box_loss",
        ):
            v = row.get(k)
            if isinstance(v, (int, float)):
                final[k] = float(v)

    weights_dir = dst / "weights"
    best_mb = round((weights_dir / "best.pt").stat().st_size / 1e6, 2) \
        if (weights_dir / "best.pt").exists() else None
    last_mb = round((weights_dir / "last.pt").stat().st_size / 1e6, 2) \
        if (weights_dir / "last.pt").exists() else None

    payload = {
        "run_name": dst.name,
        "model": model,
        "config_path": "report_generation reference (unknown pipeline config)",
        "config_hash": None,
        "dataset": {
            "version": "reference",
            "sha256": None,
            "counts": None,
        },
        "env": {
            "note": "Reference team training; env not captured. Val metrics are on "
                    "the reference team's split, not our v1 80/10/10 test split.",
        },
        "train": {
            "started_at": None,
            "ended_at": None,
            "wall_time_sec": wall,
            "epochs_requested": None,
            "epochs_completed": epochs,
            "early_stopped": None,
            "peak_gpu_mem_mb": None,
            "final_metrics_val": final or None,
            "resumed": False,
        },
        "test": None,
        "weights": {
            "best_mb": best_mb,
            "last_mb": last_mb,
            "params_m": None,
        },
        "reference_import": {
            "imported_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "note": "Imported from report_generation/. Test metrics blank — re-run "
                    "pipeline.evaluate.run() against v1 test split for apples-to-apples.",
        },
    }
    (dst / "run_meta.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", type=Path, default=Path("report_generation"))
    ap.add_argument("--out", type=Path, default=Path("runs"))
    ap.add_argument("--overwrite", action="store_true",
                    help="Overwrite existing runs/<model>_reference/ folders")
    args = ap.parse_args()

    src: Path = args.src.resolve()
    out: Path = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)

    imported = []
    for variant in VARIANTS:
        train_dir = _find_train_dir(src, variant)
        if train_dir is None:
            print(f"[import] SKIP {variant}: no train dir with results.csv found under {src / _VARIANT_TO_DIR[variant]}")
            continue
        dst = out / f"{variant}_reference"
        if dst.exists() and not args.overwrite:
            print(f"[import] SKIP {variant}: {dst} already exists (use --overwrite to replace)")
            continue
        if dst.exists():
            shutil.rmtree(dst)

        print(f"[import] {variant}: {train_dir}  ->  {dst}")
        has_best, epochs, wall = _copy_run(train_dir, dst)
        if not has_best:
            print(f"[import] WARN {variant}: no weights/best.pt copied")
        _synth_run_meta(dst, variant, epochs, wall)
        imported.append((variant, epochs, wall))

    print("")
    print("Imported reference runs:")
    for v, e, w in imported:
        print(f"  {v}_reference: {e} epochs, {w}s wall time")
    if imported:
        print("")
        print("Next: train the 4 YOLOv12 variants on Colab, download runs from Drive,")
        print("place them under runs/, then:")
        print("    python analysis/build_report.py --runs-dir runs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
