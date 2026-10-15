from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from hloc_bridge import workspace_from_rig_sfm_dir
from hloc_bridge.features import extract_global_features
from hloc_bridge.triangulation import TriangulationValidationReport
from hloc_bridge.runtime import configure_runtime
from hloc_bridge.config import HlocRuntimeConfig

from .analysis import KeyframeGraphSummary, summarize_keyframe_pairs, write_triangulation_report
from .config import FeaturePairsConfig
from .images import materialize_rig_images
from .reference import build_reference_colmap
from .rig_io import RigKeyframeRecord, load_keyframes_json


@dataclass
class FeaturePairsResult:
    keyframes: list[RigKeyframeRecord]
    image_names: list[str]
    graph: KeyframeGraphSummary
    validation: Optional[TriangulationValidationReport]
    model: Optional[object]
    paths: FeaturePairsConfig


def run_feature_pairs_pipeline(
    config: FeaturePairsConfig | None = None,
) -> FeaturePairsResult:
    """
    End-to-end from ``rig_pairs`` output:

    1. Materialize ``images/`` (if needed)
    2. Extract features + match ``pairs-all.txt``
    3. Build INS ``reference/`` + triangulation validation
    4. Global metric SfM in ``sfm/`` + ``triangulation_report.json``
    """
    cfg = config or FeaturePairsConfig()
    paths = cfg.paths

    if not paths.keyframes_json.is_file():
        raise FileNotFoundError(f"Missing {paths.keyframes_json}; run rig_pairs first")
    if not paths.pairs_all.is_file():
        raise FileNotFoundError(f"Missing {paths.pairs_all}")

    configure_runtime(HlocRuntimeConfig())

    keyframes = load_keyframes_json(paths.keyframes_json)
    materialize_rig_images(
        keyframes,
        paths.images_root,
        use_symlinks=cfg.materialize_symlinks,
        skip_if_exists=cfg.skip_images_if_present,
    )

    ws = workspace_from_rig_sfm_dir(
        paths.rig_sfm_dir,
        feature_conf_name=cfg.feature_conf_name,
        matcher_conf_name=cfg.matcher_conf_name,
        overwrite_features=cfg.overwrite_features,
    )
    ws.bootstrap()

    image_names = ws.extract_and_match_pairs_file(paths.pairs_all)
    if cfg.build_global_descriptors:
        extract_global_features(
            paths.images_root,
            image_names,
            paths.global_features,
            retrieval_conf_name=cfg.retrieval_conf_name,
            overwrite=cfg.overwrite_features,
        )
    graph = summarize_keyframe_pairs([kf.ts for kf in keyframes], paths.pairs_all)

    validation: Optional[TriangulationValidationReport] = None
    model: Optional[object] = None
    sfm_summary = ""

    if cfg.build_reference:
        build_reference_colmap(
            keyframes,
            paths.reference_dir,
            ins_path=cfg.ins_path,
            ins_time_offset_us=cfg.ins_time_offset_us,
            left_camera_id=cfg.left_camera_id,
            right_camera_id=cfg.right_camera_id,
        )

    if cfg.run_triangulation and cfg.build_reference:
        model = ws.run_triangulation(paths.reference_dir, pairs_path=paths.pairs_all)
        validation = TriangulationValidationReport.from_reconstruction(model, paths.features)
        sfm_summary = model.summary()
        write_triangulation_report(
            paths.triangulation_report,
            graph,
            validation,
            sfm_summary,
        )

    return FeaturePairsResult(
        keyframes=keyframes,
        image_names=image_names,
        graph=graph,
        validation=validation,
        model=model,
        paths=cfg,
    )
