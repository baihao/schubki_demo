from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import pycolmap

from .config import DatasetConfig, FeatureMatcherConfig, HlocPipelineConfig, OutputLayout, RunFlags
from .features import extract_local_features, resolve_feature_conf
from .localization import (
    QueryPoseResult,
    localize_query_image,
    prepare_query_features_and_matches,
    registered_image_names,
)
from .mapping_frames import build_mapping_image_list
from .matching import match_image_pairs, resolve_matcher_conf
from .pairs import write_exhaustive_pairs
from .runtime import configure_runtime
from .sfm import run_incremental_sfm
from .triangulation import (
    TriangulationValidationReport,
    run_known_pose_triangulation,
    validate_features_via_triangulation,
)


def workspace_from_rig_sfm_dir(
    rig_sfm_dir: Path,
    *,
    feature_conf_name: str = "disk",
    matcher_conf_name: str = "disk+lightglue",
    overwrite_features: bool = False,
) -> HlocWorkspace:
    """Point hloc at ``rig_sfm/`` layout (``images/``, ``pairs-all.txt``, …)."""
    rig_sfm_dir = Path(rig_sfm_dir)
    cfg = HlocPipelineConfig(
        dataset=DatasetConfig(images_root=rig_sfm_dir / "images"),
        outputs=OutputLayout(
            output_root=rig_sfm_dir,
            sfm_pairs_name="pairs-all.txt",
        ),
        models=FeatureMatcherConfig(
            feature_conf_name=feature_conf_name,
            matcher_conf_name=matcher_conf_name,
        ),
        flags=RunFlags(overwrite_features=overwrite_features),
    )
    return HlocWorkspace(cfg)


@dataclass
class HlocWorkspace:
    """
    Config-driven facade aligned with ``hloc/demo hloc.ipynb``.

    Call ``bootstrap()`` once before hloc operations (torch hub + import path).
    """

    config: HlocPipelineConfig

    def bootstrap(self) -> None:
        configure_runtime(self.config.runtime)
        if self.config.flags.clear_outputs:
            shutil.rmtree(self.config.outputs.output_root, ignore_errors=True)
        self.config.outputs.output_root.mkdir(parents=True, exist_ok=True)

    @property
    def feature_conf(self) -> dict:
        return resolve_feature_conf(self.config.models.feature_conf_name)

    @property
    def matcher_conf(self) -> dict:
        return resolve_matcher_conf(self.config.models.matcher_conf_name)

    def mapping_image_list(self) -> List[str]:
        if self.config.dataset.mapping_image_list is not None:
            return list(self.config.dataset.mapping_image_list)
        sel = self.config.dataset.mapping_selection
        if sel is None:
            raise ValueError("Provide mapping_image_list or mapping_selection")
        return build_mapping_image_list(self.config.dataset.images_root, sel)

    def extract_features_for_mapping(self, image_list: Optional[List[str]] = None) -> Path:
        names = image_list or self.mapping_image_list()
        return extract_local_features(
            self.config.dataset.images_root,
            names,
            self.config.outputs.features_path(),
            feature_conf=self.feature_conf,
            overwrite=self.config.flags.overwrite_features,
        )

    def build_sfm_pairs(self, image_list: Optional[List[str]] = None) -> Path:
        names = image_list or self.mapping_image_list()
        return write_exhaustive_pairs(self.config.outputs.sfm_pairs_path(), names)

    def match_sfm_pairs(self) -> Path:
        return match_image_pairs(
            self.config.outputs.sfm_pairs_path(),
            self.config.outputs.features_path(),
            self.config.outputs.matches_path(),
            matcher_conf=self.matcher_conf,
            overwrite=self.config.flags.overwrite_features,
        )

    def extract_and_match_pairs_file(self, pairs_path: Path | None = None) -> List[str]:
        from .matching import extract_and_match_pairs_file

        pairs_path = Path(pairs_path or self.config.outputs.sfm_pairs_path())
        names, _, _ = extract_and_match_pairs_file(
            self.config.dataset.images_root,
            pairs_path,
            self.config.outputs.features_path(),
            self.config.outputs.matches_path(),
            feature_conf_name=self.config.models.feature_conf_name,
            matcher_conf_name=self.config.models.matcher_conf_name,
            overwrite=self.config.flags.overwrite_features,
        )
        return names

    def extract_and_match_mapping(self, image_list: Optional[List[str]] = None) -> None:
        names = image_list or self.mapping_image_list()
        self.extract_features_for_mapping(names)
        self.build_sfm_pairs(names)
        self.match_sfm_pairs()

    def run_incremental_mapping(
        self,
        image_list: Optional[List[str]] = None,
    ) -> pycolmap.Reconstruction:
        names = image_list or self.mapping_image_list()
        return run_incremental_sfm(
            self.config.outputs.sfm_dir(),
            self.config.dataset.images_root,
            self.config.outputs.sfm_pairs_path(),
            self.config.outputs.features_path(),
            self.config.outputs.matches_path(),
            image_list=names,
            mapper_options=self.config.incremental_sfm.mapper_options,
            skip_geometric_verification=self.config.incremental_sfm.skip_geometric_verification,
            min_match_score=self.config.incremental_sfm.min_match_score,
        )

    def run_triangulation(
        self,
        reference_model_dir: Path,
        *,
        pairs_path: Optional[Path] = None,
        sfm_dir: Optional[Path] = None,
    ) -> pycolmap.Reconstruction:
        tc = self.config.triangulation
        return run_known_pose_triangulation(
            sfm_dir or self.config.outputs.sfm_dir(),
            reference_model_dir,
            self.config.dataset.images_root,
            pairs_path or self.config.outputs.sfm_pairs_path(),
            self.config.outputs.features_path(),
            self.config.outputs.matches_path(),
            skip_geometric_verification=tc.skip_geometric_verification,
            estimate_two_view_geometries=tc.estimate_two_view_geometries,
            min_match_score=tc.min_match_score,
            geometric_verification_max_error=tc.geometric_verification_max_error,
            verbose=tc.verbose,
            mapper_options=tc.mapper_options or None,
        )

    def validate_triangulation(
        self,
        reference_model_dir: Path,
        *,
        pairs_path: Path | None = None,
    ) -> TriangulationValidationReport:
        return validate_features_via_triangulation(
            self.config.outputs.sfm_dir(),
            reference_model_dir,
            self.config.dataset.images_root,
            pairs_path or self.config.outputs.sfm_pairs_path(),
            self.config.outputs.features_path(),
            self.config.outputs.matches_path(),
            triangulation_kwargs={
                "skip_geometric_verification": self.config.triangulation.skip_geometric_verification,
                "estimate_two_view_geometries": self.config.triangulation.estimate_two_view_geometries,
                "min_match_score": self.config.triangulation.min_match_score,
                "geometric_verification_max_error": self.config.triangulation.geometric_verification_max_error,
                "verbose": self.config.triangulation.verbose,
                "mapper_options": self.config.triangulation.mapper_options or None,
            },
        )

    def localize_query(
        self,
        model: pycolmap.Reconstruction,
        query_name: str,
        *,
        mapping_reference_names: Optional[List[str]] = None,
        skip_feature_match_prep: bool = False,
    ) -> QueryPoseResult:
        refs = mapping_reference_names
        if refs is None:
            refs = self.config.localization.reference_names_for_pnp
        if refs is None:
            refs = registered_image_names(model)

        if not skip_feature_match_prep:
            prepare_query_features_and_matches(
                self.config.dataset.images_root,
                query_name,
                refs,
                self.config.outputs.features_path(),
                self.config.outputs.matches_path(),
                self.config.outputs.loc_pairs_path(),
                feature_conf_name=self.config.models.feature_conf_name,
                matcher_conf_name=self.config.models.matcher_conf_name,
                overwrite=True,
            )

        return localize_query_image(
            model,
            self.config.dataset.images_root,
            query_name,
            self.config.outputs.features_path(),
            self.config.outputs.matches_path(),
            reference_names=refs,
            localization_config=self.config.localization.localizer_config(),
        )
