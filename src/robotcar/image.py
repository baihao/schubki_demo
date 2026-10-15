# Adapted from robotcar-dataset-sdk (CC BY-NC-SA 4.0) — Python 3.

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
from colour_demosaicing import demosaicing_CFA_Bayer_bilinear as demosaic
from PIL import Image

from .camera_model import CameraModel

BAYER_STEREO = "gbrg"
BAYER_MONO = "rggb"


def load_raw_rgb(
    image_path: Path | str,
    *,
    debayer: bool = True,
) -> np.ndarray:
    """Load PNG without undistortion (still debayered for stereo)."""
    image_path = Path(image_path)
    camera = re.search(r"(stereo|mono_(left|right|rear))", str(image_path)).group(0)
    pattern = BAYER_STEREO if camera == "stereo" else BAYER_MONO
    img = Image.open(image_path)
    if debayer:
        img = demosaic(np.array(img), pattern)
    return np.asarray(img, dtype=np.uint8)


def load_undistorted_rgb(
    image_path: Path | str,
    model: CameraModel,
    *,
    debayer: bool = True,
) -> np.ndarray:
    """Debayer (if stereo) + SDK bilinear undistortion → uint8 RGB H×W×3."""
    rgb = load_raw_rgb(image_path, debayer=debayer)
    return model.undistort(rgb)
