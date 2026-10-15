#!/usr/bin/env python3
"""Quick check: calibration, one undistorted stereo pair, INS interpolation."""

from __future__ import annotations

import sys
from pathlib import Path

V2_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V2_ROOT / "src"))

from robotcar import (  # noqa: E402
    DEFAULT_INS,
    interpolate_at_image_timestamps,
    load_stereo_calibration,
    sync_stereo_pairs,
)
from robotcar.stereo import load_stereo_undistorted  # noqa: E402


def main() -> None:
    calib = load_stereo_calibration()
    print("Left pinhole:", calib.left.pinhole.as_tuple())
    print("Right pinhole:", calib.right.pinhole.as_tuple())
    print(f"Stereo baseline from G_camera_image (m): {calib.baseline_m:.4f}")
    assert calib.baseline_m > 0.2, "baseline should be ~0.24 m for RobotCar wide stereo"

    pairs = sync_stereo_pairs(calib.left.images_dir, calib.right.images_dir)
    if not pairs:
        raise SystemExit("No stereo pairs in test_data")
    ts = pairs[0][0]
    left, right = load_stereo_undistorted(ts, calib)
    print(f"Undistorted shapes @ {ts}: left {left.shape}, right {right.shape}")

    ins_ts = [ts, pairs[min(10, len(pairs) - 1)][0]]
    poses = interpolate_at_image_timestamps(ins_ts, DEFAULT_INS)
    print(f"Interpolated {len(poses)} INS-relative poses; first translation (m):")
    print(poses[0][0:3, 3])


if __name__ == "__main__":
    main()
