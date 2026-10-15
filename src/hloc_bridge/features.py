from __future__ import annotations

from pathlib import Path
from typing import List, Union

from hloc import extract_features


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
) -> Path:
    """Wrap ``hloc.extract_features.main`` for global descriptors (NetVLAD, …)."""
    conf = retrieval_conf or resolve_feature_conf(retrieval_conf_name)
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
) -> Path:
    """Wrap ``hloc.extract_features.main`` (demo notebook extract cell)."""
    conf = feature_conf or resolve_feature_conf(feature_conf_name)
    return extract_features.main(
        conf,
        Path(image_dir),
        image_list=image_list,
        feature_path=Path(feature_path),
        overwrite=overwrite,
    )
