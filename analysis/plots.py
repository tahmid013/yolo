"""Matplotlib plots for the analysis report."""
from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .collect import RunRecord


def _candidate_columns(df, *names) -> str | None:
    """Return the first column name from `names` that exists in df."""
    if df is None:
        return None
    for n in names:
        if n in df.columns:
            return n
    return None


def training_curves(records: list[RunRecord], out: Path) -> Path | None:
    """Three-panel: train box_loss, val box_loss, val mAP50 — one line per run."""
    rows = [(r.name, r.results) for r in records if r.results is not None and not r.results.empty]
    if not rows:
        return None

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), constrained_layout=True)
    for name, df in rows:
        epoch_col = _candidate_columns(df, "epoch")
        x = df[epoch_col] if epoch_col else range(1, len(df) + 1)

        tb = _candidate_columns(df, "train/box_loss")
        if tb:
            axes[0].plot(x, df[tb], label=name)
        vb = _candidate_columns(df, "val/box_loss")
        if vb:
            axes[1].plot(x, df[vb], label=name)
        m50 = _candidate_columns(df, "metrics/mAP50(B)", "metrics/mAP_0.5")
        if m50:
            axes[2].plot(x, df[m50], label=name)

    axes[0].set_title("train/box_loss"); axes[0].set_xlabel("epoch")
    axes[1].set_title("val/box_loss");   axes[1].set_xlabel("epoch")
    axes[2].set_title("val mAP@0.5");    axes[2].set_xlabel("epoch")
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def per_class_ap(records: list[RunRecord], out: Path) -> Path | None:
    """Grouped bar chart: 9 classes x N models using test-split AP50."""
    runs_with_test = [r for r in records if r.per_class and r.per_class.get("per_class")]
    if not runs_with_test:
        return None

    # Build classes index from the first run
    first = runs_with_test[0].per_class["per_class"]
    classes = [(c["id"], c["name"]) for c in first]
    classes.sort(key=lambda x: x[0])
    n_classes = len(classes)
    n_runs = len(runs_with_test)

    fig, ax = plt.subplots(figsize=(max(10, n_classes * 1.1), 5), constrained_layout=True)
    width = 0.8 / max(n_runs, 1)
    x = np.arange(n_classes)

    for i, r in enumerate(runs_with_test):
        by_id = {c["id"]: c.get("ap50") for c in r.per_class["per_class"]}
        ys = [by_id.get(cid, 0.0) or 0.0 for cid, _ in classes]
        ax.bar(x + i * width - 0.4 + width / 2, ys, width=width, label=r.name)

    ax.set_xticks(x)
    ax.set_xticklabels([n for _, n in classes], rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("test AP@0.5")
    ax.set_ylim(0, 1)
    ax.set_title("Per-class AP@0.5 on test split")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=7)

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def size_vs_map(records: list[RunRecord], out: Path) -> Path | None:
    """Scatter: params_M vs test mAP50-95 (fallback to val)."""
    pts = []
    for r in records:
        weights = r.meta.get("weights") or {}
        params = weights.get("params_m")
        test = (r.meta.get("test") or {}).get("overall") or {}
        y = test.get("mAP50_95")
        if y is None:
            fm = (r.meta.get("train") or {}).get("final_metrics_val") or {}
            y = fm.get("metrics/mAP50-95(B)") or fm.get("metrics/mAP_0.5:0.95")
        if params is None or y is None:
            continue
        try:
            pts.append((float(params), float(y), r.name))
        except (TypeError, ValueError):
            continue
    if not pts:
        return None

    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
    ax.scatter(xs, ys, s=80)
    for x, y, name in pts:
        ax.annotate(name, (x, y), xytext=(5, 5), textcoords="offset points", fontsize=7)
    ax.set_xlabel("Parameters (M)")
    ax.set_ylabel("mAP@0.5:0.95 (test, fallback val)")
    ax.set_title("Model size vs accuracy")
    ax.grid(alpha=0.3)

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def copy_confusion_matrices(records: list[RunRecord], out_dir: Path) -> list[Path]:
    """Copy per-run confusion matrix PNGs (from training plots) into out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)
    copied: list[Path] = []
    for r in records:
        # Ultralytics writes confusion_matrix.png at the run root and inside eval/.
        # Prefer the test-split eval one.
        candidates = [
            r.run_dir / "eval" / "confusion_matrix.png",
            r.run_dir / "confusion_matrix.png",
        ]
        for src in candidates:
            if src.exists():
                dst = out_dir / f"{r.name}.png"
                shutil.copy2(src, dst)
                copied.append(dst)
                break
    return copied
