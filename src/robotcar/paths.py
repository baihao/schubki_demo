"""Default paths for SchubKI ``test_data`` and the official RobotCar SDK checkout."""

from __future__ import annotations

import os
from pathlib import Path

V2_ROOT = Path(__file__).resolve().parents[2]
SCHUBKI_ROOT = V2_ROOT.parent

TEST_DATA = Path(os.environ.get("SCHUBKI_TEST_DATA", SCHUBKI_ROOT / "test_data"))

DEFAULT_RUN = "2014-05-14-13-46-12"
DEFAULT_INS = TEST_DATA / f"{DEFAULT_RUN}-gps" / "gps" / "ins.csv"
DEFAULT_GPS = TEST_DATA / f"{DEFAULT_RUN}-gps" / "gps" / "gps.csv"
DEFAULT_LEFT_IMAGES = TEST_DATA / f"{DEFAULT_RUN}-left" / "stereo" / "left"
DEFAULT_RIGHT_IMAGES = TEST_DATA / f"{DEFAULT_RUN}-right" / "stereo" / "right"
DEFAULT_CENTER_IMAGES = TEST_DATA / f"{DEFAULT_RUN}-center" / "stereo" / "centre"

# Camera clock vs INS clock (µs): t_ins = t_image - OFFSET_US (calibrate per sequence if needed).
DEFAULT_INS_TIME_OFFSET_US = 12_180_000


def _default_sdk_root() -> Path:
    env = os.environ.get("ROBOTCAR_SDK")
    if env:
        return Path(env)
    local = V2_ROOT / "third_party" / "robotcar-dataset-sdk"
    if (local / "models").is_dir():
        return local
    return local


ROBOTCAR_SDK = _default_sdk_root()
MODELS_DIR = ROBOTCAR_SDK / "models"
EXTRINSICS_DIR = ROBOTCAR_SDK / "extrinsics"
