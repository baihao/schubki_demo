from __future__ import annotations

from typing import List, Sequence

import numpy as np


def select_keyframe_indices_by_distance(
    pose_timestamps: Sequence[int],
    ins_rows_utm_rpy: Sequence[tuple],
    min_translation_m: float = 0.75,
    min_yaw_rad: float = 0.05,
) -> List[int]:
    """
    Subsample indices using INS (northing, easting, down) + yaw.

    Each ``ins_rows_utm_rpy[i]`` is ``(n, e, d, roll, pitch, yaw)`` aligned with ``pose_timestamps[i]``.
    """
    if len(pose_timestamps) != len(ins_rows_utm_rpy):
        raise ValueError("timestamps and ins rows length mismatch")
    if not pose_timestamps:
        return []

    selected = [0]
    last = np.array(ins_rows_utm_rpy[0][:3], dtype=np.float64)
    last_yaw = float(ins_rows_utm_rpy[0][5])

    for i in range(1, len(pose_timestamps)):
        cur = np.array(ins_rows_utm_rpy[i][:3], dtype=np.float64)
        dist = np.linalg.norm(cur - last)
        dyaw = abs(float(ins_rows_utm_rpy[i][5]) - last_yaw)
        if dist >= min_translation_m or dyaw >= min_yaw_rad:
            selected.append(i)
            last = cur
            last_yaw = float(ins_rows_utm_rpy[i][5])
    return selected


def select_keyframe_indices_by_time_interval(
    pose_timestamps: Sequence[int],
    interval_us: int = 1_000_000,
) -> List[int]:
    """Subsample by minimum time gap between consecutive keyframes (wall clock)."""
    if interval_us <= 0:
        raise ValueError("interval_us must be positive")
    if not pose_timestamps:
        return []
    selected = [0]
    last_ts = int(pose_timestamps[0])
    for i in range(1, len(pose_timestamps)):
        ts = int(pose_timestamps[i])
        if ts - last_ts >= interval_us:
            selected.append(i)
            last_ts = ts
    return selected
