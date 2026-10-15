from __future__ import annotations

import os
from pathlib import Path

V2_ROOT = Path(__file__).resolve().parents[2]
SCHUBKI_ROOT = V2_ROOT.parent

DEFAULT_HLOC_ROOT = V2_ROOT / "third_party" / "Hierarchical-Localization"
DEFAULT_NOTEBOOK_DIR = V2_ROOT
DEFAULT_TORCH_HUB_DIR = V2_ROOT / "model_cache" / "torch_hub"

DEFAULT_RUN = "2014-05-14-13-46-12"
DEFAULT_TEST_DATA = Path(os.environ.get("SCHUBKI_TEST_DATA", SCHUBKI_ROOT / "test_data"))
DEFAULT_LEFT_IMAGES_ROOT = DEFAULT_TEST_DATA / f"{DEFAULT_RUN}-left"
DEFAULT_RIGHT_IMAGES_ROOT = DEFAULT_TEST_DATA / f"{DEFAULT_RUN}-right"


def resolve_hloc_root() -> Path:
    env = os.environ.get("HLOC_ROOT")
    if env:
        return Path(env)
    return DEFAULT_HLOC_ROOT
