from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import pycolmap
from hloc_bridge.config import HlocRuntimeConfig, LocalizationConfig
from hloc_bridge.global_retrieval import (
    database_names_from_loc_pairs,
    ensure_map_global_descriptors,
    prepare_hierarchical_query_localization,
)
from hloc_bridge.localization import QueryPoseResult, localize_query_image
from hloc_bridge.runtime import configure_runtime

from feature_pairs.rig_io import RigKeyframeRecord, load_keyframes_json

from .config import QueryConfig, QueryImageSpec
from .ground_truth import ground_truth_cam_from_world, infer_query_camera
from .retrieval import RetrievedKeyframe, map_image_names_from_keyframes, rank_keyframes_by_matches


@dataclass
class QueryLocalizationResult:
    spec: QueryImageSpec
    query_name: str
    gt_cam_from_world: pycolmap.Rigid3d
    gt_camera: pycolmap.Camera
    retrieval: List[RetrievedKeyframe]
    predicted: QueryPoseResult
    pnp_ret: Optional[Dict[str, Any]]


def _materialize_query_image(spec: QueryImageSpec, cfg: QueryConfig) -> Path:
    src = spec.src_path(cfg)
    if not src.is_file():
        raise FileNotFoundError(f"Query source not found: {src}")
    dst = cfg.paths.images_root / spec.rel_path
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    shutil.copy2(src, dst)
    return dst


def _registered_keyframe_names(
    model: pycolmap.Reconstruction,
    keyframes: List[RigKeyframeRecord],
) -> List[str]:
    names = map_image_names_from_keyframes(keyframes, cameras="both")
    out: List[str] = []
    for name in names:
        if model.find_image_with_name(name) is not None:
            out.append(name)
    return out


def _pnp_reference_names(
    retrieval: List[RetrievedKeyframe],
    fallback: List[str],
    max_db: Optional[int],
) -> List[str]:
    if not retrieval:
        refs = list(fallback)
    else:
        refs = [r.image_name for r in retrieval]
    if max_db is not None:
        refs = refs[:max_db]
    return refs


def run_query_localization(
    spec: QueryImageSpec,
    config: QueryConfig | None = None,
) -> QueryLocalizationResult:
    cfg = config or QueryConfig()
    configure_runtime(HlocRuntimeConfig())

    if not cfg.paths.sfm_dir.is_dir():
        raise FileNotFoundError(
            f"Missing SfM map {cfg.paths.sfm_dir}; run feature_pairs first"
        )

    _materialize_query_image(spec, cfg)
    query_name = spec.rel_path

    model = pycolmap.Reconstruction(str(cfg.paths.sfm_dir))
    keyframes = load_keyframes_json(cfg.paths.keyframes_json)
    map_names = _registered_keyframe_names(model, keyframes)

    origin_ts = keyframes[0].ts
    gt_cam = ground_truth_cam_from_world(
        spec.timestamp_us,
        spec.camera,
        cfg,
        origin_timestamp_us=origin_ts,
    )
    gt_camera = infer_query_camera(cfg.paths.images_root / query_name, spec.camera)

    if cfg.refresh_map_global_descriptors:
        ensure_map_global_descriptors(
            cfg.paths.images_root,
            map_names,
            cfg.paths.global_features,
            retrieval_conf_name=cfg.retrieval_conf_name,
        )
    elif not cfg.paths.global_features.is_file():
        raise FileNotFoundError(
            f"Missing {cfg.paths.global_features}; run feature_pairs first "
            "or set refresh_map_global_descriptors=True"
        )

    query_matcher = cfg.query_matcher_conf_name or cfg.matcher_conf_name
    prepare_hierarchical_query_localization(
        cfg.paths.images_root,
        query_name,
        map_names,
        cfg.paths.features,
        cfg.paths.global_features,
        cfg.paths.matches_loc,
        cfg.paths.loc_pairs,
        retrieval_top_k=cfg.retrieval_top_k,
        feature_conf_name=cfg.feature_conf_name,
        matcher_conf_name=query_matcher,
        retrieval_conf_name=cfg.retrieval_conf_name,
        overwrite_local=cfg.overwrite_query_descriptors,
        overwrite_global_query=cfg.overwrite_query_descriptors,
        overwrite_matches=cfg.overwrite_loc_matches,
    )

    retrieved_db = database_names_from_loc_pairs(cfg.paths.loc_pairs, query_name)
    retrieval = rank_keyframes_by_matches(
        query_name,
        retrieved_db,
        cfg.paths.matches_loc,
        top_k=cfg.retrieval_top_k,
        keyframes=keyframes,
    )

    ref_names = _pnp_reference_names(
        retrieval,
        map_names,
        cfg.pnp_max_db_images,
    )

    loc_cfg = LocalizationConfig(ransac_max_error=cfg.ransac_max_error)
    predicted = localize_query_image(
        model,
        cfg.paths.images_root,
        query_name,
        cfg.paths.features,
        cfg.paths.matches_loc,
        reference_names=ref_names,
        localization_config=loc_cfg.localizer_config(),
        query_camera=gt_camera,
        infer_camera=False,
    )

    pnp_ret = predicted.log.get("PnP_ret") if predicted.log else None

    return QueryLocalizationResult(
        spec=spec,
        query_name=query_name,
        gt_cam_from_world=gt_cam,
        gt_camera=gt_camera,
        retrieval=retrieval,
        predicted=predicted,
        pnp_ret=pnp_ret,
    )
