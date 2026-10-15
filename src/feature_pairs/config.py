from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from robotcar.paths import DEFAULT_INS, DEFAULT_INS_TIME_OFFSET_US


@dataclass
class FeaturePairsPaths:
    """Standard ``rig_sfm`` workspace (see RIG_KNOWN_POSE_TRIANGULATION §12)."""

    rig_sfm_dir: Path = field(default_factory=lambda: Path("outputs/rig_sfm"))

    @property
    def keyframes_json(self) -> Path:
        return self.rig_sfm_dir / "keyframes.json"

    @property
    def pairs_all(self) -> Path:
        return self.rig_sfm_dir / "pairs-all.txt"

    @property
    def images_root(self) -> Path:
        return self.rig_sfm_dir / "images"

    @property
    def features(self) -> Path:
        return self.rig_sfm_dir / "features.h5"

    @property
    def matches(self) -> Path:
        return self.rig_sfm_dir / "matches.h5"

    @property
    def global_features(self) -> Path:
        return self.rig_sfm_dir / "global_features.h5"

    @property
    def reference_dir(self) -> Path:
        return self.rig_sfm_dir / "reference"

    @property
    def sfm_dir(self) -> Path:
        return self.rig_sfm_dir / "sfm"

    @property
    def triangulation_report(self) -> Path:
        return self.rig_sfm_dir / "triangulation_report.json"


@dataclass
class FeaturePairsConfig:
    paths: FeaturePairsPaths = field(default_factory=FeaturePairsPaths)
    ins_path: Path = field(default_factory=lambda: DEFAULT_INS)
    ins_time_offset_us: int = DEFAULT_INS_TIME_OFFSET_US
    feature_conf_name: str = "disk"
    matcher_conf_name: str = "disk+lightglue"
    retrieval_conf_name: str = "netvlad"
    build_global_descriptors: bool = True
    overwrite_features: bool = False
    materialize_symlinks: bool = True
    skip_images_if_present: bool = True
    build_reference: bool = True
    run_triangulation: bool = True
    left_camera_id: int = 1
    right_camera_id: int = 2
