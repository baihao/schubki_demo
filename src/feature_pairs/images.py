from __future__ import annotations

import os
import shutil
from pathlib import Path

from .rig_io import RigKeyframeRecord


def materialize_rig_images(
    keyframes: list[RigKeyframeRecord],
    images_root: Path,
    *,
    use_symlinks: bool = True,
    skip_if_exists: bool = True,
) -> None:
    """
    Populate ``images/left|right/{ts}.png`` for hloc (§4 RIG doc).

    Sources come from ``rig_pairs`` ``keyframes.json`` ``*_src`` paths.
    """
    images_root = Path(images_root)
    for kf in keyframes:
        for rel_name, src in (
            (kf.left_name, kf.left_src),
            (kf.right_name, kf.right_src),
        ):
            dst = images_root / rel_name
            if skip_if_exists and dst.is_file():
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            if use_symlinks:
                if dst.is_symlink() or dst.exists():
                    dst.unlink()
                os.symlink(src.resolve(), dst)
            else:
                shutil.copy2(src, dst)
