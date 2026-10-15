from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from robotcar.paths import (
    DEFAULT_INS,
    DEFAULT_INS_TIME_OFFSET_US,
    DEFAULT_LEFT_IMAGES,
    DEFAULT_RIGHT_IMAGES,
)


@dataclass
class KeyframeConfig:
    """How to subsample synced stereo timestamps before building pairs."""

    left_dir: Path = field(default_factory=lambda: DEFAULT_LEFT_IMAGES)
    right_dir: Path = field(default_factory=lambda: DEFAULT_RIGHT_IMAGES)
    ins_path: Path = field(default_factory=lambda: DEFAULT_INS)
    ins_time_offset_us: int = DEFAULT_INS_TIME_OFFSET_US
    min_translation_m: float = 0.75
    min_yaw_rad: float = 0.05
    max_keyframes: int | None = None
    start_ts: int | None = None
    segment_duration_us: int | None = None
    """If set with ``start_ts``, only timestamps in ``[start_ts, start_ts + duration)``."""
    keyframe_interval_us: int | None = None
    """If set, pick keyframes every N µs (e.g. 1_000_000 = 1 Hz) instead of INS distance."""


@dataclass
class PairsConfig:
    """hloc ``pairs-*.txt`` generation (see RIG_KNOWN_POSE_TRIANGULATION §5, §9)."""

    max_temporal_gap: int = 5
    min_baseline_m: float = 0.0
    include_cross_camera: bool = False
    cross_max_index_gap: int = 2


@dataclass
class DefaultPairsOutput:
    """Default directories (pipeline workspace may still be ``outputs/rig_sfm/``)."""

    left_dir: Path = field(default_factory=lambda: DEFAULT_LEFT_IMAGES)
    right_dir: Path = field(default_factory=lambda: DEFAULT_RIGHT_IMAGES)
    output_dir: Path = field(default_factory=lambda: Path("outputs/rig_sfm"))
