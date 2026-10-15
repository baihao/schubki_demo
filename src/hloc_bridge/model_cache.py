"""Process-lifetime cache for hloc extractors / matchers (survive across queries)."""

from __future__ import annotations

import json
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import h5py
import numpy as np
import torch
from tqdm import tqdm

from hloc import extract_features, extractors, logger, matchers
from hloc.match_features import FeaturePairsDataset, WorkQueue, find_unique_new_pairs, writer_fn
from hloc.utils.base_model import dynamic_load
from hloc.utils.io import list_h5_names
from hloc.utils.parsers import names_to_pair, parse_retrieval

_EXTRACTORS: Dict[str, Any] = {}
_MATCHERS: Dict[str, Any] = {}
_DEVICE: Optional[str] = None


def _conf_key(kind: str, conf: Dict[str, Any]) -> str:
    payload = {"kind": kind, "model": conf.get("model", conf)}
    return json.dumps(payload, sort_keys=True, default=str)


def _device() -> str:
    global _DEVICE
    if _DEVICE is None:
        from hloc.utils.device import inference_device

        _DEVICE = inference_device()
    return _DEVICE


def clear_model_cache() -> None:
    """Drop cached models (e.g. after device change)."""
    global _DEVICE
    _EXTRACTORS.clear()
    _MATCHERS.clear()
    _DEVICE = None


def cached_models_summary() -> Dict[str, int]:
    return {"extractors": len(_EXTRACTORS), "matchers": len(_MATCHERS)}


def get_extractor(conf: Dict[str, Any]):
    key = _conf_key("extractor", conf)
    if key not in _EXTRACTORS:
        device = _device()
        Model = dynamic_load(extractors, conf["model"]["name"])
        model = Model(conf["model"]).eval().to(device)
        _EXTRACTORS[key] = model
        logger.info("Cached extractor %s on %s", conf["model"]["name"], device)
    return _EXTRACTORS[key], _device()


def get_matcher(conf: Dict[str, Any]):
    key = _conf_key("matcher", conf)
    if key not in _MATCHERS:
        device = _device()
        Model = dynamic_load(matchers, conf["model"]["name"])
        model = Model(conf["model"]).eval().to(device)
        _MATCHERS[key] = model
        logger.info("Cached matcher %s on %s", conf["model"]["name"], device)
    return _MATCHERS[key], _device()


@torch.no_grad()
def extract_features_cached(
    conf: Dict[str, Any],
    image_dir: Path,
    *,
    image_list: Optional[Union[Path, List[str]]] = None,
    feature_path: Optional[Path] = None,
    export_dir: Optional[Path] = None,
    as_half: bool = True,
    overwrite: bool = False,
) -> Path:
    """Same as ``hloc.extract_features.main`` but reuses the loaded model."""
    logger.info("Extracting features (cached model):\n%s", conf.get("model", conf))

    dataset = extract_features.ImageDataset(image_dir, conf["preprocessing"], image_list)
    if feature_path is None:
        if export_dir is None:
            raise ValueError("Provide feature_path or export_dir")
        feature_path = Path(export_dir, conf["output"] + ".h5")
    feature_path = Path(feature_path)
    feature_path.parent.mkdir(exist_ok=True, parents=True)
    skip_names = set(
        list_h5_names(feature_path) if feature_path.exists() and not overwrite else ()
    )
    dataset.names = [n for n in dataset.names if n not in skip_names]
    if len(dataset.names) == 0:
        logger.info("Skipping the extraction.")
        return feature_path

    model, device = get_extractor(conf)
    loader = torch.utils.data.DataLoader(
        dataset, num_workers=0, shuffle=False, pin_memory=False
    )
    for idx, data in enumerate(tqdm(loader)):
        name = dataset.names[idx]
        pred = model({"image": data["image"].to(device, non_blocking=True)})
        pred = {k: v[0].cpu().numpy() for k, v in pred.items()}

        pred["image_size"] = original_size = data["original_size"][0].numpy()
        if "keypoints" in pred:
            size = np.array(data["image"].shape[-2:][::-1])
            scales = (original_size / size).astype(np.float32)
            pred["keypoints"] = (pred["keypoints"] + 0.5) * scales[None] - 0.5
            if "scales" in pred:
                pred["scales"] *= scales.mean()
            uncertainty = getattr(model, "detection_noise", 1) * scales.mean()

        if as_half:
            for k in pred:
                dt = pred[k].dtype
                if (dt == np.float32) and (dt != np.float16):
                    pred[k] = pred[k].astype(np.float16)

        with h5py.File(str(feature_path), "a", libver="latest") as fd:
            if name in fd:
                del fd[name]
            grp = fd.create_group(name)
            for k, v in pred.items():
                grp.create_dataset(k, data=v)
            if "keypoints" in pred:
                grp["keypoints"].attrs["uncertainty"] = uncertainty

    logger.info("Finished exporting features.")
    return feature_path


@torch.no_grad()
def match_features_cached(
    conf: Dict[str, Any],
    pairs_path: Path,
    features_path: Path,
    matches_path: Path,
    *,
    features_ref: Optional[Path] = None,
    overwrite: bool = False,
) -> Path:
    """Same as ``hloc.match_features.match_from_paths`` but reuses the loaded model."""
    logger.info("Matching features (cached model):\n%s", conf.get("model", conf))

    feature_path_q = Path(features_path)
    feature_path_ref = Path(features_ref or features_path)
    match_path = Path(matches_path)
    if not feature_path_q.exists():
        raise FileNotFoundError(f"Query feature file {feature_path_q}.")
    if not feature_path_ref.exists():
        raise FileNotFoundError(f"Reference feature file {feature_path_ref}.")
    match_path.parent.mkdir(exist_ok=True, parents=True)

    pairs = parse_retrieval(Path(pairs_path))
    pairs = [(q, r) for q, rs in pairs.items() for r in rs]
    pairs = find_unique_new_pairs(pairs, None if overwrite else match_path)
    if len(pairs) == 0:
        logger.info("Skipping the matching.")
        return match_path

    from hloc.utils.device import feature_loader_kwargs

    model, device = get_matcher(conf)
    dataset = FeaturePairsDataset(pairs, feature_path_q, feature_path_ref)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        **feature_loader_kwargs(device),
    )
    writer_queue = WorkQueue(partial(writer_fn, match_path=match_path), 5)

    for idx, data in enumerate(tqdm(loader, smoothing=0.1)):
        data = {
            k: v if k.startswith("image") else v.to(device, non_blocking=True)
            for k, v in data.items()
        }
        pred = model(data)
        pair = names_to_pair(*pairs[idx])
        writer_queue.put((pair, pred))
    writer_queue.join()
    logger.info("Finished exporting matches.")
    return match_path
