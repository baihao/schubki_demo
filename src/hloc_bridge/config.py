from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .paths import (
    DEFAULT_HLOC_ROOT,
    DEFAULT_LEFT_IMAGES_ROOT,
    DEFAULT_NOTEBOOK_DIR,
    DEFAULT_TORCH_HUB_DIR,
)


@dataclass
class MappingFrameSelection:
    """Subsample mapping frames from a stereo directory (demo notebook logic)."""

    start_stem: str = "1400075233968075"
    frame_step: int = 6
    num_frames: int = 10
    stereo_subdir: str = "stereo/left"


@dataclass
class HlocRuntimeConfig:
    """Environment setup (torch hub, hloc import path, optional chdir)."""

    hloc_root: Path = field(default_factory=lambda: DEFAULT_HLOC_ROOT)
    torch_hub_dir: Path = field(default_factory=lambda: DEFAULT_TORCH_HUB_DIR)
    chdir_to_hloc_root: bool = False
    reload_reconstruction_module: bool = True


@dataclass
class FeatureMatcherConfig:
    feature_conf_name: str = "disk"
    matcher_conf_name: str = "disk+lightglue"


@dataclass
class RunFlags:
    clear_outputs: bool = False
    overwrite_features: bool = False


@dataclass
class OutputLayout:
    """Artifact paths under ``output_root`` (absolute recommended)."""

    output_root: Path = field(default_factory=lambda: DEFAULT_NOTEBOOK_DIR / "outputs" / "demo")
    features_name: str = "features.h5"
    matches_name: str = "matches.h5"
    sfm_pairs_name: str = "pairs-all.txt"
    loc_pairs_name: str = "pairs-loc.txt"
    sfm_dir_name: str = "sfm"

    def features_path(self) -> Path:
        return self.output_root / self.features_name

    def matches_path(self) -> Path:
        return self.output_root / self.matches_name

    def sfm_pairs_path(self) -> Path:
        return self.output_root / self.sfm_pairs_name

    def loc_pairs_path(self) -> Path:
        return self.output_root / self.loc_pairs_name

    def sfm_dir(self) -> Path:
        return self.output_root / self.sfm_dir_name


@dataclass
class DatasetConfig:
    """COLMAP ``image_dir``: root containing paths like ``stereo/left/xxx.png``."""

    images_root: Path = field(default_factory=lambda: DEFAULT_LEFT_IMAGES_ROOT)
    mapping_image_list: Optional[List[str]] = None
    mapping_selection: Optional[MappingFrameSelection] = field(
        default_factory=MappingFrameSelection
    )


@dataclass
class IncrementalSfmConfig:
    mapper_options: Dict[str, Any] = field(
        default_factory=lambda: {
            "init_min_num_inliers": 30,
            "init_min_tri_angle": 1.0,
            "init_max_forward_motion": 0.9999,
            "init_max_error": 8.0,
        }
    )
    skip_geometric_verification: bool = False
    min_match_score: Optional[float] = None


@dataclass
class TriangulationConfig:
    """Known-pose triangulation + geometric verification (v2.0 / RobotCar pipeline)."""

    skip_geometric_verification: bool = False
    estimate_two_view_geometries: bool = False
    min_match_score: Optional[float] = 0.1
    geometric_verification_max_error: float = 32.0
    verbose: bool = False
    mapper_options: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LocalizationConfig:
    ransac_max_error: float = 12.0
    refine_focal_length: bool = True
    refine_extra_params: bool = True
    reference_names_for_pnp: Optional[List[str]] = None
    """If set, PnP uses these map image names; else all registered images in the model."""

    def localizer_config(self) -> Dict[str, Any]:
        return {
            "estimation": {"ransac": {"max_error": self.ransac_max_error}},
            "refinement": {
                "refine_focal_length": self.refine_focal_length,
                "refine_extra_params": self.refine_extra_params,
            },
        }


@dataclass
class HlocPipelineConfig:
    """Single object mirroring ``demo hloc.ipynb`` setup cell."""

    runtime: HlocRuntimeConfig = field(default_factory=HlocRuntimeConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    outputs: OutputLayout = field(default_factory=OutputLayout)
    models: FeatureMatcherConfig = field(default_factory=FeatureMatcherConfig)
    flags: RunFlags = field(default_factory=RunFlags)
    incremental_sfm: IncrementalSfmConfig = field(default_factory=IncrementalSfmConfig)
    triangulation: TriangulationConfig = field(default_factory=TriangulationConfig)
    localization: LocalizationConfig = field(default_factory=LocalizationConfig)
