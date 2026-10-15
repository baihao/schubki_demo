from __future__ import annotations

from pathlib import Path
from typing import List, Sequence

from hloc import pairs_from_retrieval
from hloc.utils.io import list_h5_names

from .features import extract_global_features, extract_local_features
from .matching import match_image_pairs


def ensure_map_global_descriptors(
    image_dir: Path,
    map_names: Sequence[str],
    global_features_path: Path,
    *,
    retrieval_conf_name: str = "netvlad",
    overwrite: bool = False,
) -> Path:
    """Extract NetVLAD (etc.) for map images missing from ``global_features_path``."""
    names = [n for n in map_names if n]
    if global_features_path.exists() and not overwrite:
        have = set(list_h5_names(global_features_path))
        names = [n for n in names if n not in have]
    if not names:
        return Path(global_features_path)
    return extract_global_features(
        image_dir,
        names,
        global_features_path,
        retrieval_conf_name=retrieval_conf_name,
        overwrite=overwrite,
    )


def database_names_from_loc_pairs(loc_pairs_path: Path, query_name: str) -> List[str]:
    """DB image names paired with ``query_name`` in a hloc pairs file."""
    names: List[str] = []
    with open(loc_pairs_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 2:
                continue
            q, db = parts
            if q == query_name:
                names.append(db)
    return names


def write_loc_pairs_from_global_retrieval(
    global_descriptors_path: Path,
    loc_pairs_path: Path,
    query_name: str,
    database_names: Sequence[str],
    *,
    num_matched: int,
) -> Path:
    """NetVLAD (or other global) similarity → ``pairs-loc.txt`` (hloc paper step 1)."""
    loc_pairs_path = Path(loc_pairs_path)
    loc_pairs_path.parent.mkdir(parents=True, exist_ok=True)
    pairs_from_retrieval.main(
        Path(global_descriptors_path),
        loc_pairs_path,
        int(num_matched),
        query_list=[query_name],
        db_list=list(database_names),
    )
    return loc_pairs_path


def prepare_hierarchical_query_localization(
    image_dir: Path,
    query_name: str,
    database_names: Sequence[str],
    features_path: Path,
    global_features_path: Path,
    matches_path: Path,
    loc_pairs_path: Path,
    *,
    retrieval_top_k: int = 5,
    feature_conf_name: str = "disk",
    matcher_conf_name: str = "disk+lightglue",
    retrieval_conf_name: str = "netvlad",
    overwrite_local: bool = True,
    overwrite_global_query: bool = True,
) -> Path:
    """
    hloc-style localization prep: global retrieval prior, then local match on top-k only.

    Map global descriptors should already live in ``global_features_path`` (see ``feature_pairs``).
    """
    db_names = [n for n in database_names if n != query_name]
    if not db_names:
        raise ValueError("No database images for retrieval")

    extract_global_features(
        image_dir,
        [query_name],
        global_features_path,
        retrieval_conf_name=retrieval_conf_name,
        overwrite=overwrite_global_query,
    )
    write_loc_pairs_from_global_retrieval(
        global_features_path,
        loc_pairs_path,
        query_name,
        db_names,
        num_matched=retrieval_top_k,
    )
    extract_local_features(
        image_dir,
        [query_name],
        features_path,
        feature_conf_name=feature_conf_name,
        overwrite=overwrite_local,
    )
    match_image_pairs(
        loc_pairs_path,
        features_path,
        matches_path,
        matcher_conf_name=matcher_conf_name,
        overwrite=True,
    )
    return loc_pairs_path
