from __future__ import annotations

from pathlib import Path
from typing import List, Union

from hloc import extract_features

from .model_cache import extract_features_cached


def resolve_feature_conf(name: str) -> dict:
    if name not in extract_features.confs:
        raise KeyError(f"Unknown feature conf {name!r}; choose from {list(extract_features.confs)}")
    return extract_features.confs[name]


def extract_global_features(
    image_dir: Path,
    image_list: Union[Path, List[str]],
    global_features_path: Path,
    *,
    retrieval_conf_name: str = "netvlad",
    retrieval_conf: dict | None = None,
    overwrite: bool = False,
    use_model_cache: bool = True,
) -> Path:
    """Wrap hloc feature extract for global descriptors (NetVLAD, …)."""
    conf = retrieval_conf or resolve_feature_conf(retrieval_conf_name)
    if use_model_cache:
        return extract_features_cached(
            conf,
            Path(image_dir),
            image_list=image_list,
            feature_path=Path(global_features_path),
            overwrite=overwrite,
        )
    return extract_features.main(
        conf,
        Path(image_dir),
        image_list=image_list,
        feature_path=Path(global_features_path),
        overwrite=overwrite,
    )


def extract_local_features(
    image_dir: Path,
    image_list: Union[Path, List[str]],
    feature_path: Path,
    *,
    feature_conf_name: str = "disk",
    feature_conf: dict | None = None,
    overwrite: bool = False,
    use_model_cache: bool = True,
) -> Path:
    """Wrap hloc local feature extract; caches model for the process lifetime."""
    conf = feature_conf or resolve_feature_conf(feature_conf_name)
    if use_model_cache:
        return extract_features_cached(
            conf,
            Path(image_dir),
            image_list=image_list,
            feature_path=Path(feature_path),
            overwrite=overwrite,
        )
    return extract_features.main(
        conf,
        Path(image_dir),
        image_list=image_list,
        feature_path=Path(feature_path),
        overwrite=overwrite,
    )
