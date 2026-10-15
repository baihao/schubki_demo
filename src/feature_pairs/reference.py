from __future__ import annotations

from pathlib import Path

import numpy as np
import pycolmap
from PIL import Image
from robotcar import (
    cam_from_world,
    interpolate_at_image_timestamps,
    load_stereo_calibration,
    world_T_camera,
)

from .rig_io import RigKeyframeRecord


def _image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as im:
        w, h = im.size
    return int(w), int(h)


def _add_pinhole_camera(
    rec: pycolmap.Reconstruction,
    camera_id: int,
    width: int,
    height: int,
    pinhole,
) -> None:
    cam = pycolmap.Camera(
        camera_id=camera_id,
        model="PINHOLE",
        width=width,
        height=height,
        params=[
            float(pinhole.fx),
            float(pinhole.fy),
            float(pinhole.cx),
            float(pinhole.cy),
        ],
    )
    rec.add_camera_with_trivial_rig(cam)


def build_reference_colmap(
    keyframes: list[RigKeyframeRecord],
    output_dir: Path,
    *,
    ins_path: Path,
    ins_time_offset_us: int,
    left_camera_id: int = 1,
    right_camera_id: int = 2,
) -> Path:
    """
    INS + 外参 → ``reference/``（有位姿、无 3D 点），方案 B（§7 RIG doc）。
    """
    if not keyframes:
        raise ValueError("No keyframes for reference model")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    calib = load_stereo_calibration()
    width, height = _image_size(keyframes[0].left_src)

    rec = pycolmap.Reconstruction()
    _add_pinhole_camera(rec, left_camera_id, width, height, calib.left.pinhole)
    _add_pinhole_camera(rec, right_camera_id, width, height, calib.right.pinhole)

    timestamps = [kf.ts for kf in keyframes]
    ins_poses = interpolate_at_image_timestamps(
        timestamps,
        ins_path,
        origin_timestamp=timestamps[0],
        offset_us=ins_time_offset_us,
    )

    next_id = 1
    for kf, t_world_ins in zip(keyframes, ins_poses):
        for cal, name, cam_id in (
            (calib.left, kf.left_name, left_camera_id),
            (calib.right, kf.right_name, right_camera_id),
        ):
            t_world_cam = world_T_camera(t_world_ins, cal)
            t_cam_world = cam_from_world(t_world_cam)
            pose = pycolmap.Rigid3d(np.asarray(t_cam_world, dtype=np.float64)[:3, :4])
            image = pycolmap.Image(name=name, camera_id=cam_id, image_id=next_id)
            image.frame_id = next_id
            rec.add_image_with_trivial_frame(image, pose)
            next_id += 1

    rec.write(str(output_dir))
    return output_dir
