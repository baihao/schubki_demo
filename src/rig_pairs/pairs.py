from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Sequence, Set, Tuple

import numpy as np

from robotcar import interpolate_at_image_timestamps

from .config import KeyframeConfig, PairsConfig
from .keyframes import LEFT_PREFIX, RIGHT_PREFIX, Keyframe

Pair = Tuple[str, str]


@dataclass
class PairLists:
    stereo: List[Pair] = field(default_factory=list)
    temporal_left: List[Pair] = field(default_factory=list)
    temporal_right: List[Pair] = field(default_factory=list)
    cross: List[Pair] = field(default_factory=list)
    all_pairs: List[Pair] = field(default_factory=list)


def _positions_for_keyframes(
    keyframes: Sequence[Keyframe],
    kf_config: KeyframeConfig,
) -> np.ndarray:
    ts = [kf.ts for kf in keyframes]
    poses = interpolate_at_image_timestamps(
        ts,
        kf_config.ins_path,
        origin_timestamp=ts[0],
        offset_us=kf_config.ins_time_offset_us,
    )
    return np.array([p[0:3, 3] for p in poses], dtype=np.float64)


def stereo_pairs(keyframes: Sequence[Keyframe]) -> List[Pair]:
    return [(kf.left_name, kf.right_name) for kf in keyframes]


def temporal_pairs_for_camera(
    keyframes: Sequence[Keyframe],
    camera_prefix: str,
    max_gap: int,
    positions: np.ndarray | None,
    min_baseline_m: float,
) -> List[Pair]:
    out: List[Pair] = []
    n = len(keyframes)
    for i in range(n):
        name_i = (
            keyframes[i].left_name
            if camera_prefix == LEFT_PREFIX
            else keyframes[i].right_name
        )
        for g in range(1, max_gap + 1):
            j = i + g
            if j >= n:
                break
            if min_baseline_m > 0 and positions is not None:
                if np.linalg.norm(positions[j] - positions[i]) < min_baseline_m:
                    continue
            name_j = (
                keyframes[j].left_name
                if camera_prefix == LEFT_PREFIX
                else keyframes[j].right_name
            )
            out.append((name_i, name_j))
    return out


def cross_camera_pairs(
    keyframes: Sequence[Keyframe],
    max_index_gap: int,
) -> List[Pair]:
    out: List[Pair] = []
    n = len(keyframes)
    for i in range(n):
        for g in range(1, max_index_gap + 1):
            j = i + g
            if j >= n:
                break
            out.append((keyframes[i].left_name, keyframes[j].right_name))
            out.append((keyframes[i].right_name, keyframes[j].left_name))
    return out


def merge_pairs(*lists: Iterable[Pair]) -> List[Pair]:
    seen: Set[Tuple[str, str]] = set()
    merged: List[Pair] = []
    for pairs in lists:
        for a, b in pairs:
            key = (a, b) if a <= b else (b, a)
            if key in seen:
                continue
            seen.add(key)
            merged.append((a, b))
    return merged


def build_pair_lists(
    keyframes: Sequence[Keyframe],
    pairs_config: PairsConfig | None = None,
    keyframe_config: KeyframeConfig | None = None,
) -> PairLists:
    pc = pairs_config or PairsConfig()
    kc = keyframe_config or KeyframeConfig()

    positions = None
    if pc.min_baseline_m > 0:
        positions = _positions_for_keyframes(keyframes, kc)

    stereo = stereo_pairs(keyframes)
    t_left = temporal_pairs_for_camera(
        keyframes, LEFT_PREFIX, pc.max_temporal_gap, positions, pc.min_baseline_m
    )
    t_right = temporal_pairs_for_camera(
        keyframes, RIGHT_PREFIX, pc.max_temporal_gap, positions, pc.min_baseline_m
    )
    cross: List[Pair] = []
    if pc.include_cross_camera:
        cross = cross_camera_pairs(keyframes, pc.cross_max_index_gap)

    all_pairs = merge_pairs(stereo, t_left, t_right, cross)
    return PairLists(
        stereo=stereo,
        temporal_left=t_left,
        temporal_right=t_right,
        cross=cross,
        all_pairs=all_pairs,
    )


def write_pairs(path: Path, pairs: Sequence[Pair]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for a, b in pairs:
            f.write(f"{a} {b}\n")


def write_pair_files(output_dir: Path, lists: PairLists) -> dict[str, Path]:
    output_dir = Path(output_dir)
    paths = {
        "stereo": output_dir / "pairs-stereo.txt",
        "temporal_left": output_dir / "pairs-temporal-left.txt",
        "temporal_right": output_dir / "pairs-temporal-right.txt",
        "cross": output_dir / "pairs-cross.txt",
        "all": output_dir / "pairs-all.txt",
    }
    write_pairs(paths["stereo"], lists.stereo)
    write_pairs(paths["temporal_left"], lists.temporal_left)
    write_pairs(paths["temporal_right"], lists.temporal_right)
    write_pairs(paths["cross"], lists.cross)
    write_pairs(paths["all"], lists.all_pairs)
    return paths
