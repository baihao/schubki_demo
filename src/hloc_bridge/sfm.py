from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import pycolmap
from hloc import reconstruction


def run_incremental_sfm(
    sfm_dir: Path,
    image_dir: Path,
    pairs_path: Path,
    features_path: Path,
    matches_path: Path,
    *,
    image_list: Optional[List[str]] = None,
    mapper_options: Optional[Dict[str, Any]] = None,
    skip_geometric_verification: bool = False,
    min_match_score: Optional[float] = None,
) -> pycolmap.Reconstruction:
    """Wrap ``hloc.reconstruction.main`` (demo incremental mapping cell)."""
    model = reconstruction.main(
        Path(sfm_dir),
        Path(image_dir),
        Path(pairs_path),
        Path(features_path),
        Path(matches_path),
        image_list=image_list,
        mapper_options=mapper_options,
        skip_geometric_verification=skip_geometric_verification,
        min_match_score=min_match_score,
    )
    if model is None:
        raise RuntimeError(f"Incremental SfM failed; see colmap logs in {sfm_dir}")
    return model
