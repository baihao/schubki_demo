# `query` — 单张 query 定位

对 **`test_data`** 中某一 timestamp（左/右目）作为 query，在 **`feature_pairs`** 产出的 **`sfm/`** 上定位，返回 **GT 位姿**、**检索 keyframes**、**预测位姿**。

**流程（hloc 论文）：** 地图 **NetVLAD**（`global_features.h5`，建图时已算）→ query **NetVLAD** → top‑k 检索 → **局部匹配**（默认 DISK + LightGlue）→ PnP。

Query 专用匹配写入 **`matches_loc.h5`**，不再改写建图用的 **`matches.h5`**。

## 依赖

- 已运行 `rig_pairs` + `feature_pairs`（`sfm/`、`global_features.h5` 存在）
- `source scripts/env.sh`

## CLI

```bash
# 默认（精度优先）
python scripts/run_query.py -d outputs/rig_sfm --ts 1400075229218675 --camera left

# 速度优先：k=3、互最近邻匹配、复用已缓存的 query 特征/匹配
python scripts/run_query.py -d outputs/rig_sfm --ts 1400075229218675 --fast

# 强制重算 query 特征与 loc 匹配
python scripts/run_query.py -d outputs/rig_sfm --ts 1400075229218675 --force
```

## 加速（按收益大致排序）

| 手段 | 说明 |
|------|------|
| **`QueryConfig.fast()`** / **`--fast`** | `retrieval_top_k=3`，`query_matcher_conf_name="NN-mutual"`（仍用建图时的 DISK 描述子） |
| **减小 `retrieval_top_k`** | 局部匹配次数 ≈ k；论文里 10–20 是精度折中，demo 可 3–5 |
| **`overwrite_query_descriptors=False`**（默认） | 同一 query 图重复跑时跳过 NetVLAD + DISK |
| **`overwrite_loc_matches=False`**（默认） | pairs 未变时跳过 LightGlue/NN |
| **`refresh_map_global_descriptors=False`**（默认） | query 时不再给全地图重算 NetVLAD |
| **Query 用 `NN-mutual`，建图仍用 `disk+lightglue`** | 通过 `query_matcher_conf_name` 与 `matcher_conf_name` 分离 |
| **硬件** | CUDA / MPS；NetVLAD 权重缓存在 `model_cache/torch_hub` |

**模型缓存（默认开启）：** `hloc_bridge.features` / `matching` 经 `model_cache` 复用 NetVLAD、DISK、Matcher；同一 Python 进程 / Jupyter kernel 下多帧 query **不会每帧重新 load**。重启 kernel 后首次仍会加载一次。可 `from hloc_bridge.model_cache import clear_model_cache, cached_models_summary`。

## API

```python
from pathlib import Path
from query import QueryConfig, QueryImageSpec, QueryWorkspacePaths, run_query_localization

paths = QueryWorkspacePaths(rig_sfm_dir=Path("outputs/rig_sfm"))
result = run_query_localization(
    QueryImageSpec(timestamp_us=1400075229218675, camera="left"),
    QueryConfig.fast(paths),
)
```

## 返回值

| 字段 | 含义 |
|------|------|
| `gt_cam_from_world` / `gt_camera` | INS + 外参 GT |
| `retrieval` | NetVLAD prior 内的 keyframe，按 loc 匹配数重排 |
| `predicted` | PnP + refinement |
| `pnp_ret` | 原始 PnP 字典（notebook 可视化） |
