from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pycolmap
from hloc import pairs_from_exhaustive
from hloc.localize_sfm import QueryLocalizer, pose_from_cluster

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
) -> QueryPoseResult:
    """
    PnP + refinement for one query (demo notebook ``pose_from_cluster`` path).

    ``reference_names``: map image names to use for 2D–3D correspondences
    (demo uses the mapping ``references`` list, not necessarily all registered images).
    """
    if infer_camera and query_camera is None:
        query_camera = pycolmap.infer_camera_from_image(Path(images_root) / query_name)

    if reference_names is None:
        ref_ids = list(model.reg_image_ids())
    else:
        ref_ids = []
        for name in reference_names:
            image = model.find_image_with_name(name)
            if image is None:
                raise ValueError(f"Reference image {name!r} not in reconstruction")
            ref_ids.append(image.image_id)

    conf = localization_config or {
        "estimation": {"ransac": {"max_error": 12}},
        "refinement": {"refine_focal_length": True, "refine_extra_params": True},
    }
    localizer = QueryLocalizer(model, conf)
    ret, log = pose_from_cluster(
        localizer,
        query_name,
        query_camera,
        ref_ids,
        Path(features_path),
        Path(matches_path),
    )

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
