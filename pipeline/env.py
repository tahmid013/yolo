"""Environment capture + GPU sanity check for the Colab pipeline."""
from __future__ import annotations

import platform
import subprocess


def assert_gpu() -> None:
    """Raise RuntimeError if no CUDA GPU is available."""
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "No CUDA GPU detected. In Colab: Runtime -> Change runtime type -> GPU."
        )


def _git_commit(cwd: str | None = None) -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=cwd, stderr=subprocess.DEVNULL
        )
        return out.decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def capture_env(git_cwd: str | None = None) -> dict:
    """Snapshot the runtime environment for run_meta.json."""
    info: dict = {
        "python": platform.python_version(),
        "platform": platform.platform(),
    }

    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda"] = torch.version.cuda
        info["cudnn"] = torch.backends.cudnn.version()
        if torch.cuda.is_available():
            idx = torch.cuda.current_device()
            props = torch.cuda.get_device_properties(idx)
            info["gpu_name"] = props.name
            info["gpu_mem_gb"] = round(props.total_memory / 1e9, 2)
        else:
            info["gpu_name"] = None
            info["gpu_mem_gb"] = None
    except ImportError:
        info["torch"] = None

    try:
        import ultralytics

        info["ultralytics"] = ultralytics.__version__
    except ImportError:
        info["ultralytics"] = None

    commit = _git_commit(git_cwd)
    if commit:
        info["git_commit"] = commit
    return info
