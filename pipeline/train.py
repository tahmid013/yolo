"""Train a YOLO model and persist the run folder to Drive."""
from __future__ import annotations

import datetime as dt
import json
import shutil
import time
from pathlib import Path

from . import meta as meta_mod
from . import paths as paths_mod
from . import persist as persist_mod
from .env import capture_env


def _utc_timestamp() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d-%H%M%S")


def _find_latest_resumable(drive_runs_dir: Path, model: str) -> Path | None:
    """Find the most recent Drive run for `model` with a valid weights/last.pt."""
    candidates = []
    for sub in drive_runs_dir.iterdir():
        if not sub.is_dir() or not sub.name.startswith(f"{model}_"):
            continue
        if not (sub / "weights" / "last.pt").exists():
            continue
        candidates.append(sub)
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.name, reverse=True)
    return candidates[0]


def _register_checkpoint_callback(
    y, local_dir: Path, drive_run_dir: Path, every: int
) -> None:
    """Register a per-epoch callback that syncs to Drive every `every` epochs.

    Ultralytics fires `on_fit_epoch_end` after each epoch's train + val loops.
    We sync weights/last.pt, results.csv, args.yaml, run_meta.json — enough for
    pipeline.train.run(resume=True) to continue after a disconnect. Errors are
    caught and logged so a Drive hiccup never kills training.
    """
    if every <= 0:
        return

    def _cb(trainer):
        epoch = int(getattr(trainer, "epoch", 0)) + 1  # 1-indexed for display
        if epoch % every != 0:
            return
        try:
            print(f"[checkpoint] epoch {epoch}: sync -> {drive_run_dir}")
            persist_mod.sync_checkpoint(local_dir, drive_run_dir)
        except Exception as e:
            print(f"[checkpoint] WARN epoch {epoch}: {e}")

    try:
        y.add_callback("on_fit_epoch_end", _cb)
    except Exception as e:
        print(f"[checkpoint] WARN: could not register callback ({e}). "
              "Training will run without periodic sync.")


def _load_dataset_meta(data_yaml: Path) -> dict:
    """Read the dataset.meta.json next to data.yaml (one level up on Colab)."""
    # The unzipped layout puts data.yaml at /content/dataset/data.yaml.
    # The dataset.meta.json sits on Drive, not in the zip, so we don't read it
    # here — train.run is given the values it needs via the trainer call site.
    return {}


def run(
    model: str,
    config: str,
    *,
    run_name: str | None = None,
    drive_runs_dir: str | Path = paths_mod.RUNS_DIR,
    local_runs_dir: str | Path = paths_mod.LOCAL_RUNS,
    data_yaml: str | Path = paths_mod.LOCAL_DATASET / "data.yaml",
    dataset_meta_path: str | Path | None = None,
    base_config: str | Path = "configs/base.yaml",
    resume: bool = False,
    checkpoint_every: int = 10,
) -> Path:
    """Train one model variant.

    Returns the Drive path of the persisted run folder.
    """
    from ultralytics import YOLO

    drive_runs_dir = Path(drive_runs_dir)
    local_runs_dir = Path(local_runs_dir)
    data_yaml = Path(data_yaml)
    config = Path(config)

    if not data_yaml.exists():
        raise FileNotFoundError(
            f"data.yaml not found at {data_yaml}. Run pipeline.dataset.ensure_dataset first."
        )

    base_cfg = meta_mod.load_yaml(base_config)
    variant_cfg = meta_mod.load_yaml(config)
    merged = meta_mod.merge_configs(base_cfg, variant_cfg)
    train_kwargs = dict(merged.get("train", {}))
    cfg_hash = meta_mod.hash_config(merged)

    drive_runs_dir.mkdir(parents=True, exist_ok=True)
    local_runs_dir.mkdir(parents=True, exist_ok=True)

    if resume:
        prev = _find_latest_resumable(drive_runs_dir, model)
        if prev is None:
            raise RuntimeError(
                f"resume=True but no resumable Drive run found for model={model} "
                f"under {drive_runs_dir}"
            )
        resolved_run_name = prev.name
        print(f"[train] resuming from Drive run: {prev}")
        local_dir = local_runs_dir / resolved_run_name
        if local_dir.exists():
            shutil.rmtree(local_dir)
        shutil.copytree(prev, local_dir)
        weights_path = local_dir / "weights" / "last.pt"
        y = YOLO(str(weights_path))
        _register_checkpoint_callback(
            y, local_dir, Path(drive_runs_dir) / resolved_run_name, checkpoint_every
        )
        # Ultralytics' resume=True uses the weights' embedded training state and
        # writes back to the same project/name folder.
        started_at = dt.datetime.now(dt.timezone.utc)
        t0 = time.perf_counter()
        try:
            import torch

            torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass
        results = y.train(resume=True)
        wall = time.perf_counter() - t0
        ended_at = dt.datetime.now(dt.timezone.utc)
    else:
        resolved_run_name = run_name or f"{model}_{_utc_timestamp()}"
        local_dir = local_runs_dir / resolved_run_name
        if local_dir.exists():
            raise FileExistsError(
                f"Local run dir {local_dir} already exists. Pick a new run_name."
            )
        print(f"[train] starting {model} -> {local_dir}")
        y = YOLO(f"{model}.pt")  # pretrained weights downloaded by Ultralytics
        _register_checkpoint_callback(
            y, local_dir, Path(drive_runs_dir) / resolved_run_name, checkpoint_every
        )
        started_at = dt.datetime.now(dt.timezone.utc)
        t0 = time.perf_counter()
        try:
            import torch

            torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass
        results = y.train(
            data=str(data_yaml),
            project=str(local_runs_dir),
            name=resolved_run_name,
            exist_ok=False,
            **train_kwargs,
        )
        wall = time.perf_counter() - t0
        ended_at = dt.datetime.now(dt.timezone.utc)

    # ----- Capture metrics -----
    peak_mb: float | None = None
    try:
        import torch

        peak_mb = round(torch.cuda.max_memory_allocated() / (1024 * 1024), 1)
    except Exception:
        pass

    epochs_completed = _epochs_completed(local_dir)
    final_metrics = _final_metrics_from_results(results) or _tail_metrics(local_dir)

    weights = local_dir / "weights"
    best_mb = round((weights / "best.pt").stat().st_size / 1e6, 2) if (weights / "best.pt").exists() else None
    last_mb = round((weights / "last.pt").stat().st_size / 1e6, 2) if (weights / "last.pt").exists() else None
    params_m = _model_params_m(y)

    dataset_meta = {}
    if dataset_meta_path is not None and Path(dataset_meta_path).exists():
        dataset_meta = json.loads(Path(dataset_meta_path).read_text(encoding="utf-8"))

    epochs_requested = int(train_kwargs.get("epochs", 0))
    early_stopped = (
        epochs_completed is not None
        and epochs_requested > 0
        and epochs_completed < epochs_requested
    )

    payload = {
        "run_name": resolved_run_name,
        "model": model,
        "config_path": str(config),
        "config_hash": cfg_hash,
        "config_resolved": merged,
        "dataset": {
            "version": dataset_meta.get("version"),
            "sha256": dataset_meta.get("zip", {}).get("sha256"),
            "counts": dataset_meta.get("counts"),
        },
        "env": capture_env(),
        "train": {
            "started_at": started_at.isoformat(),
            "ended_at": ended_at.isoformat(),
            "wall_time_sec": round(wall, 1),
            "epochs_requested": epochs_requested,
            "epochs_completed": epochs_completed,
            "early_stopped": early_stopped,
            "peak_gpu_mem_mb": peak_mb,
            "final_metrics_val": final_metrics,
            "resumed": bool(resume),
        },
        "test": None,
        "weights": {"best_mb": best_mb, "last_mb": last_mb, "params_m": params_m},
    }
    meta_mod.write_run_meta(local_dir, payload)
    print(f"[train] wrote {local_dir / meta_mod.META_NAME}")

    drive_dest = persist_mod.copy_to_drive(local_dir, drive_runs_dir)
    return drive_dest


def _epochs_completed(local_dir: Path) -> int | None:
    csv = local_dir / "results.csv"
    if not csv.exists():
        return None
    try:
        # Each non-header row = one completed epoch
        with csv.open("r", encoding="utf-8") as f:
            rows = sum(1 for _ in f) - 1
        return max(rows, 0)
    except Exception:
        return None


def _final_metrics_from_results(results) -> dict | None:
    """Try to pull the headline metrics off the Ultralytics results object."""
    if results is None:
        return None
    out: dict = {}
    # Ultralytics' results.results_dict is the friendliest path
    rd = getattr(results, "results_dict", None)
    if isinstance(rd, dict):
        for k, v in rd.items():
            try:
                out[k] = float(v)
            except (TypeError, ValueError):
                continue
    return out or None


def _tail_metrics(local_dir: Path) -> dict | None:
    """Fallback: read the last row of results.csv as a dict."""
    csv = local_dir / "results.csv"
    if not csv.exists():
        return None
    try:
        import csv as _csv

        with csv.open("r", encoding="utf-8") as f:
            rows = list(_csv.DictReader(f))
        if not rows:
            return None
        last = rows[-1]
        out: dict = {}
        for k, v in last.items():
            try:
                out[k.strip()] = float(v)
            except (TypeError, ValueError):
                continue
        return out
    except Exception:
        return None


def _model_params_m(y) -> float | None:
    try:
        n = sum(p.numel() for p in y.model.parameters())
        return round(n / 1e6, 3)
    except Exception:
        return None
