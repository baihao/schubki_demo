# `query` — 单张 query 定位

对 **`test_data`** 中某一 timestamp（左/右目）作为 query，在 **`feature_pairs`** 产出的 **`sfm/`** 上定位，返回 **GT 位姿**、**检索 keyframes**、**预测位姿**（无内置可视化；可在 notebook 里用 hloc 自行绘图）。

**定位流程（与 hloc 论文一致）：** 建图时写入地图 **NetVLAD**（`global_features.h5`）→ query 上 **global retrieval top-k** → 仅对 k 对图做 **DISK + LightGlue** → PnP。

## 依赖

- 已运行 `rig_pairs` + `feature_pairs`（`outputs/rig_sfm/sfm/` 存在）
- `hloc_bridge`、hloc submodule、torch、pycolmap（见 `v2.0/requirements.txt`）
- `source scripts/env.sh`

## CLI

```bash
python scripts/run_query.py \
  --rig-sfm-dir outputs/rig_sfm \
  --ts 1400075229218675 \
  --camera left
```

## API

```python
from pathlib import Path
from query import QueryConfig, QueryImageSpec, QueryWorkspacePaths, run_query_localization

result = run_query_localization(
    QueryImageSpec(timestamp_us=1400075229218675, camera="left"),
    QueryConfig(paths=QueryWorkspacePaths(rig_sfm_dir=Path("outputs/rig_sfm"))),
)
print(result.retrieval)
print(result.gt_cam_from_world)
print(result.predicted.num_inliers, result.predicted.cam_from_world)
```

## 返回值

| 字段 | 含义 |
|------|------|
| `gt_cam_from_world` / `gt_camera` | INS + 外参 GT（与 reference 同一世界系） |
| `retrieval` | NetVLAD prior 内的 keyframe，按局部匹配数重排 |
| `predicted` | PnP + refinement（`QueryPoseResult`） |
| `pnp_ret` | 原始 PnP 字典（含 `inlier_mask` 等，供 notebook 可视化） |
