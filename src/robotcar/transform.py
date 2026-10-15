# Adapted from robotcar-dataset-sdk (CC BY-NC-SA 4.0) — Python 3.

from __future__ import annotations

from math import atan2, cos, sin, sqrt

import numpy as np

MATRIX_MATCH_TOLERANCE = 1e-4


def build_se3_transform(xyzrpy: list[float] | np.ndarray) -> np.ndarray:
    if len(xyzrpy) != 6:
        raise ValueError("Must supply 6 values to build transform")
    se3 = np.eye(4, dtype=np.float64)
    se3[0:3, 0:3] = euler_to_so3(xyzrpy[3:6])
    se3[0:3, 3] = np.asarray(xyzrpy[0:3], dtype=np.float64)
    return se3


def euler_to_so3(rpy) -> np.ndarray:
    if len(rpy) != 3:
        raise ValueError("Euler angles must have three components")
    roll, pitch, yaw = rpy
    r_x = np.array(
        [[1, 0, 0], [0, cos(roll), -sin(roll)], [0, sin(roll), cos(roll)]],
        dtype=np.float64,
    )
    r_y = np.array(
        [[cos(pitch), 0, sin(pitch)], [0, 1, 0], [-sin(pitch), 0, cos(pitch)]],
        dtype=np.float64,
    )
    r_z = np.array(
        [[cos(yaw), -sin(yaw), 0], [sin(yaw), cos(yaw), 0], [0, 0, 1]],
        dtype=np.float64,
    )
    return r_z @ r_y @ r_x


def so3_to_euler(so3: np.ndarray) -> np.ndarray:
    if so3.shape != (3, 3):
        raise ValueError("SO3 matrix must be 3x3")
    roll = atan2(so3[2, 1], so3[2, 2])
    yaw = atan2(so3[1, 0], so3[0, 0])
    denom = sqrt(so3[0, 0] ** 2 + so3[1, 0] ** 2)
    pitch_poss = [atan2(-so3[2, 0], denom), atan2(-so3[2, 0], -denom)]
    for pitch in pitch_poss:
        r = euler_to_so3((roll, pitch, yaw))
        if np.abs(so3 - r).sum() < MATRIX_MATCH_TOLERANCE:
            return np.array([roll, pitch, yaw], dtype=np.float64)
    raise ValueError("Could not find valid pitch angle")


def so3_to_quaternion(so3: np.ndarray) -> np.ndarray:
    if so3.shape != (3, 3):
        raise ValueError("SO3 matrix must be 3x3")
    r_xx, r_xy, r_xz = so3[0, 0], so3[0, 1], so3[0, 2]
    r_yx, r_yy, r_yz = so3[1, 0], so3[1, 1], so3[1, 2]
    r_zx, r_zy, r_zz = so3[2, 0], so3[2, 1], so3[2, 2]
    try:
        w = sqrt(so3.trace() + 1) / 2
    except ValueError:
        w = 0.0
    x = sqrt(max(1 + r_xx - r_yy - r_zz, 0)) / 2
    y = sqrt(max(1 + r_yy - r_xx - r_zz, 0)) / 2
    z = sqrt(max(1 + r_zz - r_yy - r_xx, 0)) / 2
    max_index = max(range(4), key=[w, x, y, z].__getitem__)
    if max_index == 0:
        x = (r_zy - r_yz) / (4 * w)
        y = (r_xz - r_zx) / (4 * w)
        z = (r_yx - r_xy) / (4 * w)
    elif max_index == 1:
        w = (r_zy - r_yz) / (4 * x)
        y = (r_xy + r_yx) / (4 * x)
        z = (r_zx + r_xz) / (4 * x)
    elif max_index == 2:
        w = (r_xz - r_zx) / (4 * y)
        x = (r_xy + r_yx) / (4 * y)
        z = (r_yz + r_zy) / (4 * y)
    else:
        w = (r_yx - r_xy) / (4 * z)
        x = (r_zx + r_xz) / (4 * z)
        y = (r_yz + r_zy) / (4 * z)
    return np.array([w, x, y, z], dtype=np.float64)


def se3_to_components(se3: np.ndarray) -> np.ndarray:
    if se3.shape != (4, 4):
        raise ValueError("SE3 transform must be a 4x4 matrix")
    xyzrpy = np.empty(6, dtype=np.float64)
    xyzrpy[0:3] = se3[0:3, 3]
    xyzrpy[3:6] = so3_to_euler(se3[0:3, 0:3])
    return xyzrpy
