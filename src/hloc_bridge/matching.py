from __future__ import annotations

from pathlib import Path
from typing import List, Union

from hloc import match_features

from .model_cache import match_features_cached


def resolve_matcher_conf(name: str) -> dict:
    if name not in match_features.confs:
        raise KeyError(
            f"Unknown matcher conf {name!r}; choose from {list(match_features.confs)}"
        )
    return match_features.confs[name]


def match_image_pairs(
    pairs_path: Path,
    features_path: Path,
    matches_path: Path,
    *,
    matcher_conf_name: str = "disk+lightglue",
    matcher_conf: dict | None = None,
    overwrite: bool = False,
    use_model_cache: bool = True,
) -> Path:
    """Wrap hloc matching; caches matcher for the process lifetime."""
    conf = matcher_conf or resolve_matcher_conf(matcher_conf_name)
    if use_model_cache:
        return match_features_cached(
            conf,
            Path(pairs_path),
            Path(features_path),
            Path(matches_path),
            overwrite=overwrite,
        )
    return match_features.main(
        conf,
        Path(pairs_path),
        features=Path(features_path),
        matches=Path(matches_path),
        overwrite=overwrite,
    )


def extract_and_match(
    image_dir: Path,
    image_list: Union[Path, List[str]],
    pairs_path: Path,
    features_path: Path,
    matches_path: Path,
    *,
    feature_conf_name: str = "disk",
    matcher_conf_name: str = "disk+lightglue",
    overwrite: bool = False,
) -> tuple[Path, Path]:
    from .features import extract_local_features

    extract_local_features(
        image_dir,
        image_list,
        features_path,
        feature_conf_name=feature_conf_name,
        overwrite=overwrite,
    )
    match_image_pairs(
        pairs_path,
        features_path,
        matches_path,
        matcher_conf_name=matcher_conf_name,
        overwrite=overwrite,
    )
    return features_path, matches_path


def extract_and_match_pairs_file(
    image_dir: Path,
    pairs_path: Path,
    features_path: Path,
    matches_path: Path,
    *,
    feature_conf_name: str = "disk",
    matcher_conf_name: str = "disk+lightglue",
    overwrite: bool = False,
) -> tuple[List[str], Path, Path]:
    """Extract features for all images in ``pairs_path``, then match listed pairs."""
    from .pairs import unique_image_names_from_pairs

    image_names = unique_image_names_from_pairs(pairs_path)
    extract_and_match(
        image_dir,
        image_names,
        pairs_path,
        features_path,
        matches_path,
        feature_conf_name=feature_conf_name,
        matcher_conf_name=matcher_conf_name,
        overwrite=overwrite,
    )
    return image_names, features_path, matches_path
