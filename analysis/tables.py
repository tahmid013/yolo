"""Build the headline comparison table from RunRecords."""
from __future__ import annotations

import pandas as pd

from .collect import RunRecord


def _val_metric(record: RunRecord, *keys: str) -> float | None:
    fm = (record.meta.get("train") or {}).get("final_metrics_val") or {}
    for k in keys:
        if k in fm and fm[k] is not None:
            try:
                return float(fm[k])
            except (TypeError, ValueError):
                continue
    return None


def _test_metric(record: RunRecord, key: str) -> float | None:
    test = record.meta.get("test") or {}
    overall = test.get("overall") or {}
    v = overall.get(key)
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def build_comparison(records: list[RunRecord]) -> pd.DataFrame:
    rows = []
    for r in records:
        train = r.meta.get("train") or {}
        weights = r.meta.get("weights") or {}
        dataset = r.meta.get("dataset") or {}
        env = r.meta.get("env") or {}

        wall = train.get("wall_time_sec")
        train_min = round(wall / 60.0, 1) if isinstance(wall, (int, float)) else None

        rows.append({
            "run_name": r.name,
            "model": r.meta.get("model"),
            "params_M": weights.get("params_m"),
            "epochs_done": train.get("epochs_completed"),
            "epochs_req": train.get("epochs_requested"),
            "train_min": train_min,
            "val_mAP50":    _val_metric(r, "metrics/mAP50(B)", "metrics/mAP_0.5"),
            "val_mAP50_95": _val_metric(r, "metrics/mAP50-95(B)", "metrics/mAP_0.5:0.95"),
            "test_mAP50":   _test_metric(r, "mAP50"),
            "test_mAP50_95": _test_metric(r, "mAP50_95"),
            "precision":    _test_metric(r, "precision") or _val_metric(r, "metrics/precision(B)"),
            "recall":       _test_metric(r, "recall") or _val_metric(r, "metrics/recall(B)"),
            "best_pt_MB":   weights.get("best_mb"),
            "config_hash":  r.meta.get("config_hash"),
            "dataset_sha":  (dataset.get("sha256") or "")[:12] if dataset.get("sha256") else None,
            "ultralytics":  env.get("ultralytics"),
            "gpu":          env.get("gpu_name"),
        })
    return pd.DataFrame(rows)


def to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "_(no runs found)_\n"
    # Replace NaN with empty string so cells like `nan` become blank in the
    # rendered markdown. tabulate + floatfmt would otherwise print "nan".
    pretty = df.copy()
    for col in pretty.columns:
        if pd.api.types.is_float_dtype(pretty[col]):
            pretty[col] = pretty[col].map(
                lambda v: "" if pd.isna(v) else f"{v:.3f}"
            )
    pretty = pretty.fillna("")
    return pretty.to_markdown(index=False)
