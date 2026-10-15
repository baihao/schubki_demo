#!/usr/bin/env python3
"""CLI: generate hloc pair files from test_data (see src/rig_pairs/README.md)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

V2_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V2_ROOT / "src"))

from rig_pairs.config import KeyframeConfig, PairsConfig  # noqa: E402
from rig_pairs.pipeline import generate_pairs_for_test_data  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description="Generate rig SfM pair lists for test_data")
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=V2_ROOT / "outputs" / "rig_sfm",
        help="Output directory for pairs-*.txt and keyframes.json",
    )
    p.add_argument("--max-keyframes", type=int, default=None)
    p.add_argument("--min-translation-m", type=float, default=0.75)
    p.add_argument("--min-yaw-rad", type=float, default=0.05)
    p.add_argument("--max-temporal-gap", type=int, default=5)
    p.add_argument("--min-baseline-m", type=float, default=0.75)
    p.add_argument("--cross-camera", action="store_true")
    args = p.parse_args()

    kc = KeyframeConfig(
        min_translation_m=args.min_translation_m,
        min_yaw_rad=args.min_yaw_rad,
        max_keyframes=args.max_keyframes,
    )
    pc = PairsConfig(
        max_temporal_gap=args.max_temporal_gap,
        min_baseline_m=args.min_baseline_m,
        include_cross_camera=args.cross_camera,
    )
    keyframes, lists, paths = generate_pairs_for_test_data(args.output, kc, pc)
    print(f"Keyframes: {len(keyframes)}")
    print(f"Pairs (all): {len(lists.all_pairs)}")
    for name, path in paths.items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
