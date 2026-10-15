from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .config import HlocRuntimeConfig


def ensure_hloc_importable(hloc_root: Path) -> None:
    root = Path(hloc_root).resolve()
    if not (root / "hloc").is_dir():
        raise FileNotFoundError(f"hloc package not found under {root}")
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)


def setup_torch_hub(cache_dir: Path) -> None:
    import torch

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    torch.hub.set_dir(str(cache_dir))


def inference_device_label(*, allow_mps: bool = True) -> str:
    """Log-friendly device string after ``configure_runtime`` has put hloc on ``sys.path``."""
    from hloc.utils.device import inference_device

    return inference_device(allow_mps=allow_mps)


def configure_runtime(config: HlocRuntimeConfig) -> None:
    ensure_hloc_importable(config.hloc_root)
    setup_torch_hub(config.torch_hub_dir)
    if config.chdir_to_hloc_root:
        os.chdir(config.hloc_root)
    if config.reload_reconstruction_module:
        import hloc.reconstruction

        importlib.reload(hloc.reconstruction)
