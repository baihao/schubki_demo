from __future__ import annotations

from pathlib import Path
from typing import List

from .config import MappingFrameSelection


def build_mapping_image_list(
    images_root: Path,
    selection: MappingFrameSelection,
) -> List[str]:
    """
    Build hloc-style relative paths (e.g. ``stereo/left/{ts}.png``).

    Frames are taken at ``start_stem + k * frame_step`` in sorted filename order.
    """
    stereo_dir = Path(images_root) / Path(selection.stereo_subdir)
    if not stereo_dir.is_dir():
        raise FileNotFoundError(f"Stereo directory not found: {stereo_dir}")

    frames = sorted(p.name for p in stereo_dir.glob("*.png"))
    start_name = f"{selection.start_stem}.png"
    if start_name not in frames:
        raise ValueError(f"start_stem {selection.start_stem} not found in {stereo_dir}")

    start_idx = frames.index(start_name)
    rel_prefix = Path(selection.stereo_subdir)
    out: List[str] = []
    for k in range(selection.num_frames):
        idx = start_idx + selection.frame_step * k
        if idx >= len(frames):
            break
        out.append(str(rel_prefix / frames[idx]))
    if not out:
        raise ValueError("No mapping frames selected; check step / num_frames")
    return out
