"""Read/write run_meta.json — the per-run source of truth for analysis."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

META_NAME = "run_meta.json"

# Repo root = parent of the `pipeline/` package. Used to resolve short relative
# paths like "configs/base.yaml" independently of the process CWD.
REPO_ROOT = Path(__file__).resolve().parent.parent


def hash_config(cfg: dict, length: int = 12) -> str:
    """Stable hash of a config dict (canonical JSON, sorted keys)."""
    canonical = json.dumps(cfg, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:length]


def resolve_repo_path(path: str | Path) -> Path:
    """Resolve a relative path against REPO_ROOT; return absolute paths as-is."""
    p = Path(path)
    if p.is_absolute() or p.exists():
        return p
    return REPO_ROOT / p


def load_yaml(path: str | Path) -> dict:
    p = resolve_repo_path(path)
    return yaml.safe_load(p.read_text(encoding="utf-8")) or {}


def merge_configs(base: dict, override: dict) -> dict:
    """Deep merge `override` into a copy of `base`. Override wins on conflicts."""
    out: dict = json.loads(json.dumps(base))  # cheap deep copy of JSON-safe dicts
    _deep_update(out, override)
    return out


def _deep_update(dst: dict, src: dict) -> None:
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_update(dst[k], v)
        else:
            dst[k] = v


def write_run_meta(run_dir: str | Path, payload: dict) -> Path:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    out = run_dir / META_NAME
    out.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return out


def update_run_meta(run_dir: str | Path, patch: dict) -> Path:
    run_dir = Path(run_dir)
    out = run_dir / META_NAME
    current: dict[str, Any] = {}
    if out.exists():
        current = json.loads(out.read_text(encoding="utf-8"))
    _deep_update(current, patch)
    out.write_text(json.dumps(current, indent=2, default=str), encoding="utf-8")
    return out


def read_run_meta(run_dir: str | Path) -> dict:
    return json.loads((Path(run_dir) / META_NAME).read_text(encoding="utf-8"))
