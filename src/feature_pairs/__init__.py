"""hloc features, matching, triangulation, and global SfM from ``rig_pairs`` outputs."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .config import FeaturePairsConfig, FeaturePairsPaths

if TYPE_CHECKING:
    from .pipeline import FeaturePairsResult

_LAZY = {
    "FeaturePairsResult": (".pipeline", "FeaturePairsResult"),
    "run_feature_pairs_pipeline": (".pipeline", "run_feature_pairs_pipeline"),
    "build_reference_colmap": (".reference", "build_reference_colmap"),
}


def __getattr__(name: str):
    if name not in _LAZY:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    mod_name, attr = _LAZY[name]
    mod = importlib.import_module(mod_name, __package__)
    return getattr(mod, attr)


__all__ = [
    "FeaturePairsConfig",
    "FeaturePairsPaths",
    "FeaturePairsResult",
    "build_reference_colmap",
    "run_feature_pairs_pipeline",
]
