from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List

from hloc_bridge import read_pairs, unique_image_names_from_pairs


@dataclass
class KeyframeGraphSummary:
    num_keyframes: int
    num_images: int
    num_pairs: int
    num_stereo_pairs: int
    num_temporal_pairs: int
    image_names: List[str] = field(default_factory=list)


def summarize_keyframe_pairs(
    keyframe_ts: List[int],
    pairs_path: Path,
) -> KeyframeGraphSummary:
    pairs = read_pairs(pairs_path)
    names = unique_image_names_from_pairs(pairs_path)
    stereo = 0
    temporal = 0
    for a, b in pairs:
        side_a, ts_a = a.split("/", 1)
        side_b, ts_b = b.split("/", 1)
        stem_a = Path(ts_a).stem
        stem_b = Path(ts_b).stem
        if stem_a == stem_b and side_a != side_b:
            stereo += 1
        elif side_a == side_b and stem_a != stem_b:
            temporal += 1
    return KeyframeGraphSummary(
        num_keyframes=len(keyframe_ts),
        num_images=len(names),
        num_pairs=len(pairs),
        num_stereo_pairs=stereo,
        num_temporal_pairs=temporal,
        image_names=names,
    )


def write_triangulation_report(
    path: Path,
    graph: KeyframeGraphSummary,
    validation: Any,
    sfm_summary: str,
) -> None:
    per_image = [
        {
            "name": row.name,
            "num_keypoints": row.num_keypoints,
            "num_triangulated": row.num_triangulated,
            "ratio": row.triangulation_ratio,
        }
        for row in validation.per_image
    ]
    payload: Dict[str, Any] = {
        "graph": asdict(graph),
        "triangulation": {
            "num_points3D": validation.num_points3D,
            "num_observations": validation.num_observations,
            "mean_track_length": validation.mean_track_length,
            "mean_reprojection_error": validation.mean_reprojection_error,
            "per_image": per_image,
            "summary_text": validation.summary_text,
        },
        "sfm_summary": sfm_summary,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
