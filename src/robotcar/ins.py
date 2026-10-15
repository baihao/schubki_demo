# Adapted from robotcar-dataset-sdk (CC BY-NC-SA 4.0) — Python 3.

from __future__ import annotations

import bisect
import csv
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np

from .paths import DEFAULT_INS, DEFAULT_INS_TIME_OFFSET_US
from .transform import build_se3_transform, so3_to_quaternion


def image_timestamp_to_ins_timestamp(
    image_timestamp_us: int,
    offset_us: int = DEFAULT_INS_TIME_OFFSET_US,
) -> int:
    """Map camera timestamp to INS clock: t_ins = t_image - offset."""
    return int(image_timestamp_us - offset_us)


def interpolate_ins_poses(
    ins_path: Path | str,
    pose_timestamps: Sequence[int],
    origin_timestamp: int,
    *,
    use_rtk: bool = False,
    require_good: bool = True,
) -> List[np.ndarray]:
    """
    Interpolate SE(3) poses from RobotCar ``ins.csv`` (UTM northing/easting/down + RPY).

    Returns poses **relative to** ``origin_timestamp`` (4×4 ``T_world←ins`` in metres/radians).
    """
    pose_timestamps = list(pose_timestamps)
    ins_timestamps: List[int] = []
    abs_poses: List[np.ndarray] = []

    upper_timestamp = max(max(pose_timestamps), origin_timestamp)

    with open(ins_path, newline="") as ins_file:
        reader = csv.reader(ins_file)
        next(reader)
        for row in reader:
            if require_good and row[1] != "INS_SOLUTION_GOOD":
                continue
            timestamp = int(row[0])
            ins_timestamps.append(timestamp)
            utm = row[5:8] if not use_rtk else row[4:7]
            rpy = row[-3:] if not use_rtk else row[11:14]
            xyzrpy = [float(v) for v in utm] + [float(v) for v in rpy]
            abs_poses.append(build_se3_transform(xyzrpy))
            if timestamp >= upper_timestamp:
                break

    if not ins_timestamps:
        raise ValueError(f"No INS rows loaded from {ins_path}")

    return interpolate_poses(
        ins_timestamps, abs_poses, pose_timestamps, origin_timestamp
    )


def interpolate_poses(
    pose_timestamps: Sequence[int],
    abs_poses: Sequence[np.ndarray],
    requested_timestamps: Sequence[int],
    origin_timestamp: int,
) -> List[np.ndarray]:
    requested = [origin_timestamp, *list(requested_timestamps)]
    requested_arr = np.array(requested, dtype=np.int64)
    pose_timestamps_arr = np.array(pose_timestamps, dtype=np.int64)

    if len(pose_timestamps) != len(abs_poses):
        raise ValueError("Must supply same number of timestamps as poses")

    abs_quaternions = np.zeros((4, len(abs_poses)))
    abs_positions = np.zeros((3, len(abs_poses)))
    for i, pose in enumerate(abs_poses):
        if i > 0 and pose_timestamps[i - 1] >= pose_timestamps[i]:
            raise ValueError("Pose timestamps must be in ascending order")
        abs_quaternions[:, i] = so3_to_quaternion(pose[0:3, 0:3])
        abs_positions[:, i] = pose[0:3, 3]

    upper_indices = [bisect.bisect(pose_timestamps_arr, pt) for pt in requested_arr]
    lower_indices = [u - 1 for u in upper_indices]
    upper_indices = [min(i, len(pose_timestamps) - 1) for i in upper_indices]

    denom = pose_timestamps_arr[upper_indices] - pose_timestamps_arr[lower_indices]
    denom = np.maximum(denom, 1)
    fractions = (requested_arr - pose_timestamps_arr[lower_indices]) / denom

    quaternions_lower = abs_quaternions[:, lower_indices]
    quaternions_upper = abs_quaternions[:, upper_indices]
    d_array = (quaternions_lower * quaternions_upper).sum(0)

    linear_interp_indices = np.nonzero(d_array >= 1)[0]
    sin_interp_indices = np.nonzero(d_array < 1)[0]

    scale0_array = np.zeros(d_array.shape)
    scale1_array = np.zeros(d_array.shape)
    scale0_array[linear_interp_indices] = 1 - fractions[linear_interp_indices]
    scale1_array[linear_interp_indices] = fractions[linear_interp_indices]

    theta_array = np.arccos(np.abs(d_array[sin_interp_indices]))
    scale0_array[sin_interp_indices] = (
        np.sin((1 - fractions[sin_interp_indices]) * theta_array) / np.sin(theta_array)
    )
    scale1_array[sin_interp_indices] = (
        np.sin(fractions[sin_interp_indices] * theta_array) / np.sin(theta_array)
    )
    negative_d_indices = np.nonzero(d_array < 0)[0]
    scale1_array[negative_d_indices] = -scale1_array[negative_d_indices]

    quaternions_interp = np.tile(scale0_array, (4, 1)) * quaternions_lower + np.tile(
        scale1_array, (4, 1)
    ) * quaternions_upper

    positions_lower = abs_positions[:, lower_indices]
    positions_upper = abs_positions[:, upper_indices]
    positions_interp = np.multiply(
        np.tile((1 - fractions), (3, 1)), positions_lower
    ) + np.multiply(np.tile(fractions, (3, 1)), positions_upper)

    n = len(requested)
    poses_mat = np.zeros((4, 4 * n), dtype=np.float64)
    qw, qx, qy, qz = quaternions_interp

    poses_mat[0, 0::4] = 1 - 2 * qy**2 - 2 * qz**2
    poses_mat[0, 1::4] = 2 * (qx * qy - qz * qw)
    poses_mat[0, 2::4] = 2 * (qx * qz + qy * qw)
    poses_mat[1, 0::4] = 2 * (qx * qy + qz * qw)
    poses_mat[1, 1::4] = 1 - 2 * qx**2 - 2 * qz**2
    poses_mat[1, 2::4] = 2 * (qy * qz - qx * qw)
    poses_mat[2, 0::4] = 2 * (qx * qz - qy * qw)
    poses_mat[2, 1::4] = 2 * (qy * qz + qx * qw)
    poses_mat[2, 2::4] = 1 - 2 * qx**2 - 2 * qy**2
    poses_mat[0:3, 3::4] = positions_interp
    poses_mat[3, 3::4] = 1

    poses_mat = np.linalg.solve(poses_mat[0:4, 0:4], poses_mat)

    return [poses_mat[0:4, i * 4 : (i + 1) * 4].copy() for i in range(1, n)]


def interpolate_at_image_timestamps(
    image_timestamps_us: Sequence[int],
    ins_path: Path | str | None = None,
    *,
    origin_timestamp: int | None = None,
    offset_us: int = DEFAULT_INS_TIME_OFFSET_US,
    **kwargs,
) -> List[np.ndarray]:
    """
    Convenience: convert image µs → INS clock, then interpolate ``T_world←ins``.

    ``origin_timestamp`` defaults to first image's INS time.
    """
    ins_path = Path(ins_path or DEFAULT_INS)
    image_timestamps_us = list(image_timestamps_us)
    if not image_timestamps_us:
        return []
    ins_ts = [image_timestamp_to_ins_timestamp(t, offset_us) for t in image_timestamps_us]
    origin = (
        image_timestamp_to_ins_timestamp(origin_timestamp, offset_us)
        if origin_timestamp is not None
        else ins_ts[0]
    )
    return interpolate_ins_poses(ins_path, ins_ts, origin, **kwargs)
