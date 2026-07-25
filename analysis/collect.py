"""Discover and load per-run artifacts from a `runs/` directory."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass
class RunRecord:
    run_dir: Path
    name: str
    meta: dict
    results: pd.DataFrame | None  # epoch-by-epoch metrics
    per_class: dict | None        # test-split per-class metrics


def load_runs(runs_dir: str | Path) -> list[RunRecord]:
    runs_dir = Path(runs_dir)
    if not runs_dir.is_dir():
        raise FileNotFoundError(f"runs dir not found: {runs_dir}")

    records: list[RunRecord] = []
    for sub in sorted(runs_dir.iterdir()):
        if not sub.is_dir():
            continue
        if sub.name.endswith(".tmp") or sub.name.endswith(".stale"):
            continue
        meta_path = sub / "run_meta.json"
        if not meta_path.exists():
            print(f"[collect] skipping {sub.name} (no run_meta.json)")
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))

        results_csv = sub / "results.csv"
        results_df: pd.DataFrame | None = None
        if results_csv.exists():
            try:
                results_df = pd.read_csv(results_csv)
                results_df.columns = [c.strip() for c in results_df.columns]
            except Exception as e:
                print(f"[collect] WARN: could not read {results_csv}: {e}")

        per_class_path = sub / "eval" / "per_class.json"
        per_class: dict | None = None
        if per_class_path.exists():
            per_class = json.loads(per_class_path.read_text(encoding="utf-8"))

        records.append(RunRecord(
            run_dir=sub, name=sub.name, meta=meta,
            results=results_df, per_class=per_class,
        ))

    def _sort_key(r: RunRecord) -> str:
        # ended_at may be None (imported reference runs); fall back to the run
        # name so all keys are comparable strings.
        ended = (r.meta.get("train") or {}).get("ended_at")
        return ended if isinstance(ended, str) else r.name
    records.sort(key=_sort_key)
    return records
