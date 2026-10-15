from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

from robotcar import interpolate_at_image_timestamps
from robotcar.keyframes import (
    select_keyframe_indices_by_distance,
    select_keyframe_indices_by_time_interval,
)
from robotcar.stereo import sync_stereo_pairs
from robotcar.transform import se3_to_components

from .config import KeyframeConfig

StereoPair = Tuple[int, Path, Path]
LEFT_PREFIX = "left"
RIGHT_PREFIX = "right"


@dataclass(frozen=True)
class Keyframe:
    index: int
    ts: int
    left_src: Path
    right_src: Path

    @property
    def left_name(self) -> str:
        return f"{LEFT_PREFIX}/{self.ts}.png"

    @property
    def right_name(self) -> str:
        return f"{RIGHT_PREFIX}/{self.ts}.png"


def load_synced_stereo(
    left_dir: Path,
    right_dir: Path,
    suffix: str = ".png",
) -> List[StereoPair]:
    return sync_stereo_pairs(left_dir, right_dir, suffix=suffix)


def _utm_rpy_rows_for_timestamps(
    timestamps: Sequence[int],
    ins_path: Path,
    offset_us: int,
) -> List[tuple[float, float, float, float, float, float]]:
    if not timestamps:
        return []
    poses = interpolate_at_image_timestamps(
        timestamps,
        ins_path,
        origin_timestamp=timestamps[0],
        offset_us=offset_us,
    )
    rows: List[tuple[float, float, float, float, float, float]] = []
    for p in poses:
        c = se3_to_components(p)
        rows.append(
            (float(c[0]), float(c[1]), float(c[2]), float(c[3]), float(c[4]), float(c[5]))
        )
    return rows


def select_keyframes(
    synced: Sequence[StereoPair],
    config: KeyframeConfig | None = None,
) -> List[Keyframe]:
    """Pick keyframes using INS motion (translation / yaw)."""
    cfg = config or KeyframeConfig()
    if not synced:
        raise ValueError("No synced stereo pairs")

    pairs = list(synced)
    if cfg.start_ts is not None:
        ts_list = [p[0] for p in pairs]
        if cfg.start_ts not in ts_list:
            raise ValueError(f"start_ts {cfg.start_ts} not in synced pairs")
        start = ts_list.index(cfg.start_ts)
        pairs = pairs[start:]

    if cfg.segment_duration_us is not None:
        if cfg.start_ts is None:
            raise ValueError("segment_duration_us requires start_ts")
        t_end = cfg.start_ts + cfg.segment_duration_us
        pairs = [p for p in pairs if p[0] < t_end]

    timestamps = [p[0] for p in pairs]
    if cfg.keyframe_interval_us is not None:
        indices = select_keyframe_indices_by_time_interval(
            timestamps, cfg.keyframe_interval_us
        )
    else:
        ins_rows = _utm_rpy_rows_for_timestamps(
            timestamps, cfg.ins_path, cfg.ins_time_offset_us
        )
        indices = select_keyframe_indices_by_distance(
            timestamps,
            ins_rows,
            min_translation_m=cfg.min_translation_m,
            min_yaw_rad=cfg.min_yaw_rad,
        )
    if cfg.max_keyframes is not None:
        indices = indices[: cfg.max_keyframes]

    return [
        Keyframe(
            index=k,
            ts=pairs[i][0],
            left_src=Path(pairs[i][1]),
            right_src=Path(pairs[i][2]),
        )
        for k, i in enumerate(indices)
    ]


def write_keyframes_json(path: Path, keyframes: Sequence[Keyframe]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [
        {
            "index": kf.index,
            "ts": kf.ts,
            "left_name": kf.left_name,
            "right_name": kf.right_name,
            "left_src": str(kf.left_src),
            "right_src": str(kf.right_src),
        }
        for kf in keyframes
    ]
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
