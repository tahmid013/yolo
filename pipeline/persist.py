"""Atomic copy of a local run folder to Google Drive.

Strategy: copy to `<dest_parent>/<name>.tmp/` first, then `os.rename` to
`<dest_parent>/<name>/`. If a Colab disconnect kills the copy mid-flight, only
a `.tmp` folder remains, which the next session can safely delete.
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path


def copy_to_drive(local: str | Path, drive_parent: str | Path) -> Path:
    local = Path(local)
    drive_parent = Path(drive_parent)
    if not local.is_dir():
        raise FileNotFoundError(f"Local run dir not found: {local}")

    drive_parent.mkdir(parents=True, exist_ok=True)
    final = drive_parent / local.name
    tmp = drive_parent / f"{local.name}.tmp"

    if tmp.exists():
        print(f"[persist] removing leftover {tmp}")
        shutil.rmtree(tmp)

    print(f"[persist] copying {local} -> {tmp}")
    shutil.copytree(local, tmp)

    if final.exists():
        # Update in place: move old aside and remove after rename succeeds.
        stale = drive_parent / f"{local.name}.stale"
        if stale.exists():
            shutil.rmtree(stale)
        os.rename(final, stale)
        try:
            os.rename(tmp, final)
        except OSError:
            # Roll back if rename failed
            os.rename(stale, final)
            raise
        shutil.rmtree(stale, ignore_errors=True)
    else:
        os.rename(tmp, final)

    print(f"[persist] -> {final}")
    return final


def sync_checkpoint(local_dir: str | Path, drive_dir: str | Path) -> None:
    """Snapshot the resume-critical files from local_dir to drive_dir.

    Called periodically DURING training so a disconnect doesn't lose all progress.
    Unlike copy_to_drive (which does one atomic full-folder rename at end of
    training), this does per-file atomic writes so it's cheap to call every N
    epochs. On disconnect, drive_dir holds enough state for
    pipeline.train.run(resume=True) to continue from the last synced epoch.
    """
    local_dir = Path(local_dir)
    drive_dir = Path(drive_dir)
    drive_dir.mkdir(parents=True, exist_ok=True)

    for name in ("weights", "results.csv", "args.yaml", "run_meta.json"):
        src = local_dir / name
        if not src.exists():
            continue
        dst = drive_dir / name
        if src.is_dir():
            dst.mkdir(exist_ok=True)
            for f in src.iterdir():
                if not f.is_file():
                    continue
                tmp = dst / (f.name + ".tmp")
                shutil.copy2(f, tmp)
                os.replace(tmp, dst / f.name)  # atomic
        else:
            tmp = dst.with_name(dst.name + ".tmp")
            shutil.copy2(src, tmp)
            os.replace(tmp, dst)


def cleanup_tmp(drive_parent: str | Path) -> int:
    """Remove leftover *.tmp / *.stale folders from previous failed copies."""
    drive_parent = Path(drive_parent)
    removed = 0
    for sub in drive_parent.iterdir():
        if sub.is_dir() and (sub.name.endswith(".tmp") or sub.name.endswith(".stale")):
            print(f"[persist] cleaning {sub}")
            shutil.rmtree(sub, ignore_errors=True)
            removed += 1
    return removed
