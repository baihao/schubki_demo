from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
from hloc.utils.io import get_matches

from feature_pairs.rig_io import RigKeyframeRecord


@dataclass(frozen=True)
class RetrievedKeyframe:
    image_name: str
    match_count: int
    mean_score: float
    keyframe_index: int | None = None
    timestamp_us: int | None = None


def map_image_names_from_keyframes(
    keyframes: Sequence[RigKeyframeRecord],
    *,
    cameras: str = "both",
) -> List[str]:
    names: List[str] = []
    for kf in keyframes:
        if cameras in ("both", "left"):
            names.append(kf.left_name)
        if cameras in ("both", "right"):
            names.append(kf.right_name)
    return names


def _name_to_keyframe(keyframes: Sequence[RigKeyframeRecord]) -> Dict[str, RigKeyframeRecord]:
    out: Dict[str, RigKeyframeRecord] = {}
    for kf in keyframes:
        out[kf.left_name] = kf
        out[kf.right_name] = kf
    return out


def rank_keyframes_by_matches(
    query_name: str,
    candidate_names: Sequence[str],
    matches_path: Path,
    *,
    top_k: int = 5,
    min_score: float = 0.0,
    keyframes: Sequence[RigKeyframeRecord] | None = None,
) -> List[RetrievedKeyframe]:
    """
    Re-rank **retrieval-prior** DB images by local matcher count (before PnP).

    ``candidate_names`` should be the NetVLAD top-k subset (see ``pairs-loc.txt``).
    """
    kf_by_name = _name_to_keyframe(keyframes) if keyframes else {}
    scored: List[RetrievedKeyframe] = []
    for name in candidate_names:
        if name == query_name:
            continue
        try:
            matches, scores = get_matches(matches_path, query_name, name)
        except (KeyError, OSError, ValueError):
            continue
        if len(matches) == 0:
            continue
        if min_score > 0:
            keep = scores >= min_score
            matches = matches[keep]
            scores = scores[keep]
        if len(matches) == 0:
            continue
        kf = kf_by_name.get(name)
        scored.append(
            RetrievedKeyframe(
                image_name=name,
                match_count=int(len(matches)),
                mean_score=float(np.mean(scores)) if len(scores) else 0.0,
                keyframe_index=kf.index if kf else None,
                timestamp_us=kf.ts if kf else None,
            )
        )
    scored.sort(key=lambda r: (r.match_count, r.mean_score), reverse=True)
    return scored[:top_k]
