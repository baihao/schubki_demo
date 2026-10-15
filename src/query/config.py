from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

from robotcar.paths import (
    DEFAULT_INS,
    DEFAULT_INS_TIME_OFFSET_US,
    DEFAULT_LEFT_IMAGES,
    DEFAULT_RIGHT_IMAGES,
)

CameraSide = Literal["left", "right"]


@dataclass
class QueryWorkspacePaths:
    """Paths under ``rig_sfm`` (after ``feature_pairs``)."""

    rig_sfm_dir: Path = field(default_factory=lambda: Path("outputs/rig_sfm"))

    @property
    def images_root(self) -> Path:
        return self.rig_sfm_dir / "images"

    @property
    def sfm_dir(self) -> Path:
        return self.rig_sfm_dir / "sfm"

    @property
    def features(self) -> Path:
        return self.rig_sfm_dir / "features.h5"

    @property
    def matches(self) -> Path:
        return self.rig_sfm_dir / "matches.h5"

    @property
    def keyframes_json(self) -> Path:
        return self.rig_sfm_dir / "keyframes.json"

    @property
    def loc_pairs(self) -> Path:
        return self.rig_sfm_dir / "pairs-loc.txt"

    @property
    def global_features(self) -> Path:
        return self.rig_sfm_dir / "global_features.h5"


@dataclass
class QueryConfig:
    paths: QueryWorkspacePaths = field(default_factory=QueryWorkspacePaths)
    ins_path: Path = field(default_factory=lambda: DEFAULT_INS)
    ins_time_offset_us: int = DEFAULT_INS_TIME_OFFSET_US
    feature_conf_name: str = "disk"
    matcher_conf_name: str = "disk+lightglue"
    retrieval_conf_name: str = "netvlad"
    ransac_max_error: float = 12.0
    retrieval_top_k: int = 5
    """Global retrieval prior (NetVLAD): top-k DB images for local matching."""
    pnp_max_db_images: Optional[int] = None
    """Cap map images for PnP; ``None`` = all keyframe map names in the model."""
    left_stereo_dir: Path = field(
        default_factory=lambda: DEFAULT_LEFT_IMAGES
    )
    right_stereo_dir: Path = field(
        default_factory=lambda: DEFAULT_RIGHT_IMAGES
    )


@dataclass(frozen=True)
class QueryImageSpec:
    """Query from ``test_data`` stereo PNG (timestamp µs + eye)."""

    timestamp_us: int
    camera: CameraSide = "left"

    @property
    def rel_path(self) -> str:
        return f"query/{self.camera}/{self.timestamp_us}.png"

    def src_path(self, cfg: QueryConfig) -> Path:
        root = cfg.left_stereo_dir if self.camera == "left" else cfg.right_stereo_dir
        return root / f"{self.timestamp_us}.png"
