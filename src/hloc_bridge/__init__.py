"""Wrappers around hloc for SchubKI mapping and localization pipelines."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .config import (
    DatasetConfig,
    FeatureMatcherConfig,
    HlocPipelineConfig,
    HlocRuntimeConfig,
    IncrementalSfmConfig,
    LocalizationConfig,
    MappingFrameSelection,
    OutputLayout,
    RunFlags,
    TriangulationConfig,
)
from .mapping_frames import build_mapping_image_list
from .paths import (
    DEFAULT_HLOC_ROOT,
    DEFAULT_LEFT_IMAGES_ROOT,
    DEFAULT_NOTEBOOK_DIR,
    resolve_hloc_root,
)
from .runtime import configure_runtime, ensure_hloc_importable, setup_torch_hub

if TYPE_CHECKING:
    from .localization import QueryPoseResult
    from .triangulation import TriangulationValidationReport
    from .workspace import HlocWorkspace

_LAZY_EXPORTS = {
    "HlocWorkspace": (".workspace", "HlocWorkspace"),
    "QueryPoseResult": (".localization", "QueryPoseResult"),
    "TriangulationValidationReport": (
        ".triangulation",
        "TriangulationValidationReport",
    ),
    "extract_local_features": (".features", "extract_local_features"),
    "resolve_feature_conf": (".features", "resolve_feature_conf"),
    "match_image_pairs": (".matching", "match_image_pairs"),
    "resolve_matcher_conf": (".matching", "resolve_matcher_conf"),
    "write_exhaustive_pairs": (".pairs", "write_exhaustive_pairs"),
    "write_pairs_from_file": (".pairs", "write_pairs_from_file"),
    "read_pairs": (".pairs", "read_pairs"),
    "unique_image_names_from_pairs": (".pairs", "unique_image_names_from_pairs"),
    "extract_and_match_pairs_file": (".matching", "extract_and_match_pairs_file"),
    "workspace_from_rig_sfm_dir": (".workspace", "workspace_from_rig_sfm_dir"),
    "rank_database_images_by_matches": (".retrieval", "rank_database_images_by_matches"),
    "MatchRetrievalScore": (".retrieval", "MatchRetrievalScore"),
    "run_incremental_sfm": (".sfm", "run_incremental_sfm"),
    "run_known_pose_triangulation": (
        ".triangulation",
        "run_known_pose_triangulation",
    ),
    "validate_features_via_triangulation": (
        ".triangulation",
        "validate_features_via_triangulation",
    ),
    "localize_query_image": (".localization", "localize_query_image"),
    "registered_image_names": (".localization", "registered_image_names"),
}


def __getattr__(name: str):
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    mod_name, attr = _LAZY_EXPORTS[name]
    mod = importlib.import_module(mod_name, __package__)
    return getattr(mod, attr)


__all__ = [
    "DEFAULT_HLOC_ROOT",
    "DEFAULT_LEFT_IMAGES_ROOT",
    "DEFAULT_NOTEBOOK_DIR",
    "DatasetConfig",
    "FeatureMatcherConfig",
    "HlocPipelineConfig",
    "HlocRuntimeConfig",
    "HlocWorkspace",
    "IncrementalSfmConfig",
    "LocalizationConfig",
    "MappingFrameSelection",
    "OutputLayout",
    "QueryPoseResult",
    "RunFlags",
    "TriangulationConfig",
    "TriangulationValidationReport",
    "build_mapping_image_list",
    "configure_runtime",
    "ensure_hloc_importable",
    "extract_local_features",
    "localize_query_image",
    "match_image_pairs",
    "registered_image_names",
    "resolve_feature_conf",
    "resolve_matcher_conf",
    "resolve_hloc_root",
    "run_incremental_sfm",
    "run_known_pose_triangulation",
    "setup_torch_hub",
    "validate_features_via_triangulation",
    "write_exhaustive_pairs",
    "write_pairs_from_file",
    "read_pairs",
    "unique_image_names_from_pairs",
    "extract_and_match_pairs_file",
    "workspace_from_rig_sfm_dir",
    "rank_database_images_by_matches",
    "MatchRetrievalScore",
]
