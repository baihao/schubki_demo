from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import pycolmap
from hloc import triangulation as hloc_triangulation
from hloc.utils.io import get_keypoints


@dataclass
class ImageTriangulationStat:
    name: str
    num_keypoints: int
    num_triangulated: int

    @property
    def triangulation_ratio(self) -> float:
        if self.num_keypoints == 0:
            return 0.0
        return self.num_triangulated / self.num_keypoints


@dataclass
class TriangulationValidationReport:
    """Summary after known-pose triangulation (feature → 3D association check)."""

    num_points3D: int
    num_observations: int
    mean_track_length: float
    mean_reprojection_error: float
    per_image: List[ImageTriangulationStat] = field(default_factory=list)
    summary_text: str = ""

    @classmethod
    def from_reconstruction(
        cls,
        model: pycolmap.Reconstruction,
        features_path: Path,
    ) -> TriangulationValidationReport:
        per_image: List[ImageTriangulationStat] = []
        for _image_id, image in model.images.items():
            kps = get_keypoints(features_path, image.name)
            n_kp = len(kps)
            n_tri = sum(1 for p in image.points2D if p.has_point3D())
            per_image.append(
                ImageTriangulationStat(
                    name=image.name,
                    num_keypoints=n_kp,
                    num_triangulated=n_tri,
                )
            )

        summary = model.summary()
        num_points3D = len(model.points3D)
        num_observations = sum(len(p.track.elements) for p in model.points3D.values())
        mean_track_length = (
            float(num_observations) / num_points3D if num_points3D > 0 else 0.0
        )
        mean_reproj = 0.0
        if num_points3D > 0 and hasattr(model, "compute_mean_reprojection_error"):
            mean_reproj = float(model.compute_mean_reprojection_error())
        return cls(
            num_points3D=num_points3D,
            num_observations=num_observations,
            mean_track_length=mean_track_length,
            mean_reprojection_error=mean_reproj,
            per_image=per_image,
            summary_text=summary,
        )


def run_known_pose_triangulation(
    sfm_dir: Path,
    reference_model_dir: Path,
    image_dir: Path,
    pairs_path: Path,
    features_path: Path,
    matches_path: Path,
    *,
    skip_geometric_verification: bool = False,
    estimate_two_view_geometries: bool = False,
    min_match_score: Optional[float] = None,
    geometric_verification_max_error: float = 32.0,
    verbose: bool = False,
    mapper_options: Optional[Dict[str, Any]] = None,
) -> pycolmap.Reconstruction:
    """
    Wrap ``hloc.triangulation.main``: geometric verification under fixed poses,
    then ``pycolmap.triangulate_points`` (v2.0 RIG pipeline).
    """
    return hloc_triangulation.main(
        Path(sfm_dir),
        Path(reference_model_dir),
        Path(image_dir),
        Path(pairs_path),
        Path(features_path),
        Path(matches_path),
        skip_geometric_verification=skip_geometric_verification,
        estimate_two_view_geometries=estimate_two_view_geometries,
        min_match_score=min_match_score,
        geometric_verification_max_error=geometric_verification_max_error,
        verbose=verbose,
        mapper_options=mapper_options,
    )


def validate_features_via_triangulation(
    sfm_dir: Path,
    reference_model_dir: Path,
    image_dir: Path,
    pairs_path: Path,
    features_path: Path,
    matches_path: Path,
    triangulation_kwargs: Optional[Dict[str, Any]] = None,
) -> TriangulationValidationReport:
    """Run triangulation and return per-image triangulated / keypoint counts."""
    kwargs = triangulation_kwargs or {}
    model = run_known_pose_triangulation(
        sfm_dir,
        reference_model_dir,
        image_dir,
        pairs_path,
        features_path,
        matches_path,
        **kwargs,
    )
    return TriangulationValidationReport.from_reconstruction(model, Path(features_path))
