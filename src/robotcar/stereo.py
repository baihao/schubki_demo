"""Stereo pair sync and batch undistort for ``test_data`` left/right folders."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator, List, Tuple

import numpy as np

from .calibration import StereoCalibration, load_stereo_calibration
from .image import load_undistorted_rgb
from .paths import DEFAULT_LEFT_IMAGES, DEFAULT_RIGHT_IMAGES


def image_timestamps_from_dir(image_dir: Path, suffix: str = ".png") -> List[int]:
    stems = sorted(p.stem for p in Path(image_dir).glob(f"*{suffix}"))
    return [int(s) for s in stems]


def sync_stereo_pairs(
    left_dir: Path,
    right_dir: Path,
    suffix: str = ".png",
) -> List[Tuple[int, Path, Path]]:
    left_dir = Path(left_dir)
    right_dir = Path(right_dir)
    left = {int(p.stem): p for p in left_dir.glob(f"*{suffix}")}
    right = {int(p.stem): p for p in right_dir.glob(f"*{suffix}")}
    common = sorted(set(left.keys()) & set(right.keys()))
    return [(ts, left[ts], right[ts]) for ts in common]


def load_stereo_undistorted(
    timestamp_us: int,
    calib: StereoCalibration,
    *,
    left_dir: Path | None = None,
    right_dir: Path | None = None,
    suffix: str = ".png",
) -> tuple[np.ndarray, np.ndarray]:
    left_dir = Path(left_dir or calib.left.images_dir)
    right_dir = Path(right_dir or calib.right.images_dir)
    left_path = left_dir / f"{timestamp_us}{suffix}"
    right_path = right_dir / f"{timestamp_us}{suffix}"
    left_rgb = load_undistorted_rgb(left_path, calib.left.model)
    right_rgb = load_undistorted_rgb(right_path, calib.right.model)
    return left_rgb, right_rgb


def iter_undistorted_pairs(
    calib: StereoCalibration | None = None,
    *,
    left_dir: Path | None = None,
    right_dir: Path | None = None,
    max_pairs: int | None = None,
) -> Iterator[tuple[int, np.ndarray, np.ndarray]]:
    calib = calib or load_stereo_calibration()
    left_dir = Path(left_dir or DEFAULT_LEFT_IMAGES)
    right_dir = Path(right_dir or DEFAULT_RIGHT_IMAGES)
    pairs = sync_stereo_pairs(left_dir, right_dir)
    if max_pairs is not None:
        pairs = pairs[:max_pairs]
    for ts, lp, rp in pairs:
        yield (
            ts,
            load_undistorted_rgb(lp, calib.left.model),
            load_undistorted_rgb(rp, calib.right.model),
        )
