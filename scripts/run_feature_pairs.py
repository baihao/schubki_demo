#!/usr/bin/env python3
"""Run feature extraction, matching, triangulation, and global SfM from rig_pairs output."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

V2_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V2_ROOT / "src"))

from feature_pairs import FeaturePairsConfig, FeaturePairsPaths, run_feature_pairs_pipeline  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description="hloc feature/match/triangulate from rig_pairs")
    p.add_argument(
        "-d",
        "--rig-sfm-dir",
        type=Path,
        default=V2_ROOT / "outputs" / "rig_sfm",
        help="rig_pairs workspace (keyframes.json, pairs-all.txt)",
    )
    p.add_argument("--overwrite-features", action="store_true")
    p.add_argument(
        "--no-triangulation",
        action="store_true",
        help="Only materialize images + extract + match",
    )
    p.add_argument(
        "--copy-images",
        action="store_true",
        help="Copy PNGs instead of symlinks into images/",
    )
    args = p.parse_args()

    cfg = FeaturePairsConfig(
        paths=FeaturePairsPaths(rig_sfm_dir=args.rig_sfm_dir),
        overwrite_features=args.overwrite_features,
        materialize_symlinks=not args.copy_images,
        run_triangulation=not args.no_triangulation,
        build_reference=not args.no_triangulation,
    )
    result = run_feature_pairs_pipeline(cfg)
    print(f"Keyframes: {len(result.keyframes)}")
    print(f"Images in pairs: {result.graph.num_images}, pairs: {result.graph.num_pairs}")
    print(f"Features for: {len(result.image_names)} images")
    if result.validation:
        print(f"points3D: {result.validation.num_points3D}")
        print(f"mean reproj: {result.validation.mean_reprojection_error:.4f}")
        print(f"Report: {cfg.paths.triangulation_report}")


if __name__ == "__main__":
    main()
