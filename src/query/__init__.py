"""Query localization against ``rig_sfm`` maps with GT, retrieval, and viz."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .config import QueryConfig, QueryImageSpec, QueryWorkspacePaths

if TYPE_CHECKING:
    from .pipeline import QueryLocalizationResult

_LAZY = {
    "QueryLocalizationResult": (".pipeline", "QueryLocalizationResult"),
    "run_query_localization": (".pipeline", "run_query_localization"),
    "RetrievedKeyframe": (".retrieval", "RetrievedKeyframe"),
}


def __getattr__(name: str):
    if name not in _LAZY:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    mod_name, attr = _LAZY[name]
    mod = importlib.import_module(mod_name, __package__)
    return getattr(mod, attr)


__all__ = [
    "QueryConfig",
    "QueryImageSpec",
    "QueryLocalizationResult",
    "QueryWorkspacePaths",
    "RetrievedKeyframe",
    "run_query_localization",
]
