"""PyTorch inference device selection (CUDA → MPS → CPU)."""

from __future__ import annotations

import torch


def inference_device(*, allow_mps: bool = True) -> str:
    if torch.cuda.is_available():
        return "cuda"
    if allow_mps and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def feature_loader_kwargs(device: str) -> dict:
    """DataLoader options safe for hloc feature extract / match on each device."""
    if device == "cuda":
        return {"num_workers": 5, "pin_memory": True}
    return {"num_workers": 0, "pin_memory": False}
