from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pycolmap
from hloc import pairs_from_exhaustive
from hloc.localize_sfm import (
    QueryLocalizer,
    do_covisibility_clustering,
    pose_from_cluster,
)

from .features import extract_local_features
from .global_retrieval import prepare_hierarchical_query_localization
from .matching import match_image_pairs


@dataclass
class QueryPoseResult:
    query_name: str
    success: bool
    num_inliers: int
    num_correspondences: int
    cam_from_world: Optional[pycolmap.Rigid3d]
    log: Dict[str, Any]

    def cam_from_world_matrix(self) -> Optional[np.ndarray]:
        if self.cam_from_world is None:
            return None
        return np.asarray(self.cam_from_world.matrix())


def registered_image_names(model: pycolmap.Reconstruction) -> List[str]:
    return [model.images[i].name for i in model.reg_image_ids()]


def prepare_query_features_and_matches(
    image_dir: Path,
    query_name: str,
    reference_names: List[str],
    features_path: Path,
    matches_path: Path,
    loc_pairs_path: Path,
    *,
    feature_conf_name: str = "disk",
    matcher_conf_name: str = "disk+lightglue",
    overwrite: bool = True,
) -> Path:
    """Small-scene demo prep: exhaustive local matching vs all ``reference_names``."""
    extract_local_features(
        image_dir,
        [query_name],
        features_path,
        feature_conf_name=feature_conf_name,
        overwrite=overwrite,
    )
    pairs_from_exhaustive.main(
        Path(loc_pairs_path),
        image_list=[query_name],
        ref_list=reference_names,
    )
    match_image_pairs(
        loc_pairs_path,
        features_path,
        matches_path,
        matcher_conf_name=matcher_conf_name,
        overwrite=overwrite,
    )
    return loc_pairs_path


def _reference_ids(
    model: pycolmap.Reconstruction,
    reference_names: Optional[List[str]],
) -> List[int]:
    if reference_names is None:
        return list(model.reg_image_ids())
    ref_ids: List[int] = []
    for name in reference_names:
        image = model.find_image_with_name(name)
        if image is None:
            raise ValueError(f"Reference image {name!r} not in reconstruction")
        ref_ids.append(image.image_id)
    return ref_ids


def _pose_result_from_ret(
    query_name: str,
    ret: Optional[Dict[str, Any]],
    log: Dict[str, Any],
) -> QueryPoseResult:
    if ret is None:
        return QueryPoseResult(
            query_name=query_name,
            success=False,
            num_inliers=0,
            num_correspondences=int(log.get("num_matches", 0)),
            cam_from_world=None,
            log=log,
        )
    return QueryPoseResult(
        query_name=query_name,
        success=True,
        num_inliers=int(ret["num_inliers"]),
        num_correspondences=len(ret["inlier_mask"]),
        cam_from_world=ret["cam_from_world"],
        log=log,
    )


def localize_query_image(
    model: pycolmap.Reconstruction,
    images_root: Path,
    query_name: str,
    features_path: Path,
    matches_path: Path,
    *,
    reference_names: Optional[List[str]] = None,
    localization_config: Optional[Dict[str, Any]] = None,
    infer_camera: bool = True,
    query_camera: Optional[pycolmap.Camera] = None,
    covisibility_clustering: bool = False,
) -> QueryPoseResult:
    """
    PnP + refinement for one query (hloc ``pose_from_cluster`` path).

    ``reference_names``: map image names for 2D–3D correspondences.
    If ``covisibility_clustering`` is True, split those DB frames into covisible
    connected components, run PnP per cluster, and keep the pose with most inliers.
    """
    if infer_camera and query_camera is None:
        query_camera = pycolmap.infer_camera_from_image(Path(images_root) / query_name)

    ref_ids = _reference_ids(model, reference_names)
    conf = localization_config or {
        "estimation": {"ransac": {"max_error": 12}},
        "refinement": {"refine_focal_length": True, "refine_extra_params": True},
    }
    localizer = QueryLocalizer(model, conf)
    features_path = Path(features_path)
    matches_path = Path(matches_path)

    if not covisibility_clustering or len(ref_ids) <= 1:
        ret, log = pose_from_cluster(
            localizer,
            query_name,
            query_camera,
            ref_ids,
            features_path,
            matches_path,
        )
        log = dict(log)
        log["covisibility_clustering"] = False
        log["num_clusters"] = 1
        return _pose_result_from_ret(query_name, ret, log)

    clusters = do_covisibility_clustering(ref_ids, model)
    best_inliers = -1
    best_ret: Optional[Dict[str, Any]] = None
    best_log: Dict[str, Any] = {}
    cluster_logs: List[Dict[str, Any]] = []
    best_cluster: Optional[int] = None

    for i, cluster_ids in enumerate(clusters):
        ret, log = pose_from_cluster(
            localizer,
            query_name,
            query_camera,
            cluster_ids,
            features_path,
            matches_path,
        )
        log = dict(log)
        log["cluster_index"] = i
        log["cluster_size"] = len(cluster_ids)
        cluster_logs.append(log)
        if ret is not None and int(ret["num_inliers"]) > best_inliers:
            best_inliers = int(ret["num_inliers"])
            best_ret = ret
            best_log = log
            best_cluster = i

    if best_ret is None:
        # Fall back to joint PnP over all refs if every cluster failed.
        ret, log = pose_from_cluster(
            localizer,
            query_name,
            query_camera,
            ref_ids,
            features_path,
            matches_path,
        )
        log = dict(log)
        log["covisibility_clustering"] = True
        log["num_clusters"] = len(clusters)
        log["best_cluster"] = None
        log["log_clusters"] = cluster_logs
        log["fallback_joint_pnp"] = True
        return _pose_result_from_ret(query_name, ret, log)

    best_log = dict(best_log)
    best_log["covisibility_clustering"] = True
    best_log["num_clusters"] = len(clusters)
    best_log["best_cluster"] = best_cluster
    best_log["log_clusters"] = cluster_logs
    best_log["db"] = ref_ids
    return _pose_result_from_ret(query_name, best_ret, best_log)
