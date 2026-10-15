from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import numpy as np
from hloc.utils.io import get_matches


@dataclass(frozen=True)
class MatchRetrievalScore:
    image_name: str
    match_count: int
    mean_score: float


def rank_database_images_by_matches(
    query_name: str,
    database_names: Sequence[str],
    matches_path: Path,
    *,
    top_k: int = 10,
    min_mean_score: float = 0.0,
) -> List[MatchRetrievalScore]:
    """Rank DB images by number of feature matches to ``query_name`` in ``matches.h5``."""
    ranked: List[MatchRetrievalScore] = []
    for name in database_names:
        if name == query_name:
            continue
        try:
            matches, scores = get_matches(matches_path, query_name, name)
        except (KeyError, OSError, ValueError):
            continue
        if len(matches) == 0:
            continue
        mean = float(np.mean(scores)) if len(scores) else 0.0
        if mean < min_mean_score:
            continue
        ranked.append(
            MatchRetrievalScore(
                image_name=name,
                match_count=len(matches),
                mean_score=mean,
            )
        )
    ranked.sort(key=lambda x: (x.match_count, x.mean_score), reverse=True)
    return ranked[:top_k]
