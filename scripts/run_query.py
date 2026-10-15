#!/usr/bin/env python3
"""Localize one test_data frame against rig_sfm map (see src/query/README.md)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

V2_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V2_ROOT / "src"))

from query import QueryConfig, QueryImageSpec, QueryWorkspacePaths, run_query_localization  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description="Query localization on rig_sfm map")
    p.add_argument(
        "-d",
        "--rig-sfm-dir",
        type=Path,
        default=V2_ROOT / "outputs" / "rig_sfm",
    )
    p.add_argument("--ts", type=int, required=True, help="Image timestamp (µs)")
    p.add_argument(
        "--camera",
        choices=("left", "right"),
        default="left",
    )
    p.add_argument("--retrieval-top-k", type=int, default=5)
    args = p.parse_args()

    spec = QueryImageSpec(timestamp_us=args.ts, camera=args.camera)
    cfg = QueryConfig(
        paths=QueryWorkspacePaths(rig_sfm_dir=args.rig_sfm_dir),
        retrieval_top_k=args.retrieval_top_k,
    )
    result = run_query_localization(spec, cfg)

    print(f"Query: {result.query_name}")
    print(f"GT (cam_from_world): {result.gt_cam_from_world.summary()}")
    print("Retrieval (match count):")
    for r in result.retrieval:
        print(f"  {r.image_name}: {r.match_count} matches, mean_score={r.mean_score:.3f}")
    if result.predicted.success:
        print(
            f"Predicted inliers: {result.predicted.num_inliers}/"
            f"{result.predicted.num_correspondences}"
        )
        if result.predicted.cam_from_world is not None:
            print(f"Predicted: {result.predicted.cam_from_world.summary()}")
    else:
        print("Predicted pose: FAILED")


if __name__ == "__main__":
    main()
