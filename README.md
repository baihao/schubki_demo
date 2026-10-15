# SchubKI v2.0 — Rig + 已知位姿三角化 / BA

在 **hloc 特征与匹配** 不变的前提下，用 **INS 固定位姿** 走 **pycolmap `triangulate_points`**（必要时 **BA + INS prior**），产出与 hloc demo 同构的 **`points3D.bin`** 稀疏地图：**metric 尺度、点更少、更稳**，避免增量 mapper 初始化失败。

| 文档 | 内容 |
|------|------|
| [docs/RIG_KNOWN_POSE_TRIANGULATION.md](docs/RIG_KNOWN_POSE_TRIANGULATION.md) | **实现思路**（Rig、pairs、reference 模型、三角化、BA） |
| [docs/MULTI_VIEW_MAPPING_LOCALIZATION_OPTIONS.md](docs/MULTI_VIEW_MAPPING_LOCALIZATION_OPTIONS.md) | **方案探索**：多视图建图与定位可选路线 |

---

## 仓库布局

`v2.0` 为**独立 git 仓库**；上游依赖以 **submodule** 管理：

```
v2.0/
├── third_party/
│   ├── robotcar-dataset-sdk/       # 官方 models / extrinsics
│   └── Hierarchical-Localization/  # hloc（需 SchubKI 补丁）
├── patches/hloc/
├── src/
│   ├── robotcar/       # 标定、INS、立体同步
│   ├── hloc_bridge/    # hloc 工作区、特征、三角化、query
│   ├── rig_pairs/      # keyframe + pairs
│   ├── feature_pairs/  # DISK + LightGlue + 三角化 → sfm/
│   └── query/          # NetVLAD 检索 + 局部匹配 + PnP
├── scripts/            # CLI 与 env
└── test/
    ├── pipeline_one_minute.ipynb   # 全流程 demo（20 s 片段）
    └── query_fast_demo.ipynb       # gap=3 建图 + Fast query
```

---

## 跑 Demo 前需要什么

### 1. Oxford RobotCar `test_data`

默认路径：**`../test_data`**（相对 `v2.0`，即 SchubKI 根目录下的 `test_data`）。其它位置请设置：

```bash
export SCHUBKI_TEST_DATA=/path/to/test_data
```

Demo 使用序列 **`2014-05-14-13-46-12`**，至少需要：

| 路径（相对 `test_data`） | 用途 |
|--------------------------|------|
| `2014-05-14-13-46-12-left/stereo/left/` | 左目 PNG |
| `2014-05-14-13-46-12-right/stereo/right/` | 右目 PNG |
| `2014-05-14-13-46-12-gps/gps/ins.csv` | INS 位姿（三角化 / GT） |

### 2. 机器与环境

- **Python 3.12**（推荐 macOS Apple Silicon 或带 CUDA 的 Linux）
- **磁盘**：首次跑会下载 DISK / LightGlue / **NetVLAD（约 530 MB）** 等到 `v2.0/model_cache/torch_hub`
- **macOS**：需 `KMP_DUPLICATE_LIB_OK=TRUE`（`scripts/env.sh` 已设置），避免 torch 与 pycolmap 的 OpenMP 冲突

### 3. 第三方代码

```bash
cd v2.0
git submodule update --init --recursive
bash scripts/apply_schubki_hloc_patches.sh
```

若无 submodule，可用脚本拉取并打补丁：

```bash
bash scripts/setup_robotcar_sdk.sh
bash scripts/setup_hloc.sh
```

确认 RobotCar 模型目录存在：

```bash
ls third_party/robotcar-dataset-sdk/models
```

---

## 一次性环境安装

在 **`v2.0` 目录**执行（路径相对本仓库根）：

```bash
cd v2.0
python3.12 -m venv .venv-schubki-v2
source .venv-schubki-v2/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt
```

> **注意：** 若终端同时有 conda `(base)`，不要用裸命令 `pip install`，请始终用 **`python -m pip`**，并确认 `which python` 指向 `.venv-schubki-v2/bin/python`。

可选：注册 Jupyter kernel（跑 notebook 时更方便）：

```bash
bash scripts/install_jupyter_kernel.sh
```

验证 RobotCar 读数与标定：

```bash
source scripts/env.sh
python scripts/smoke_test_robotcar.py
```

---

## Demo 方式 A：Jupyter（推荐，含 3D 可视化）

端到端 notebook：**[test/pipeline_one_minute.ipynb](test/pipeline_one_minute.ipynb)** · 快速建图+定位：**[test/query_fast_demo.ipynb](test/query_fast_demo.ipynb)**（`max_temporal_gap=3` + `QueryConfig.fast()`）

| 步骤 | 内容 |
|------|------|
| §0 | 路径、`HLOC_ROOT`、torch hub 缓存 |
| §1 | 从 `test_data` 取 **20 s** 立体时间窗 |
| §2 | `rig_pairs`：约 **每秒 1 个 keyframe**，写 `pairs-*.txt`、`keyframes.json` |
| §3 | `feature_pairs`：DISK + LightGlue + INS 三角化 → **`sfm/`**、**`global_features.h5`**，Plotly 看稀疏地图 |
| §4 | 段内选一帧 **query**：GT vs 预测位姿、相机锥 3D 图 |

**操作：**

1. `cd v2.0 && source scripts/env.sh`
2. 用 Cursor / VS Code 或 Jupyter 打开 `test/pipeline_one_minute.ipynb`
3. 选择 kernel：**SchubKI v2.0**，或解释器 **`v2.0/.venv-schubki-v2/bin/python`**
4. **Run All**（§3 最耗时：特征、匹配、三角化；§4 首次会拉 NetVLAD）

默认输出目录：

```text
v2.0/test/outputs/segment_20s_rig/
├── keyframes.json
├── pairs-all.txt
├── images/              # 参与建图的 PNG（符号链接或拷贝）
├── features.h5
├── matches.h5
├── global_features.h5   # query 全局检索用
├── sfm/                 # COLMAP 模型（points3D.bin 等）
└── triangulation_report.json
```

Notebook 从 **`test/`** 作为 cwd 运行（§0 里 `V2_ROOT = TEST_DIR.parent`）。

---

## Demo 方式 B：命令行（全序列或自定义输出目录）

每次新开终端建议：

```bash
cd v2.0
source scripts/env.sh
```

### 步骤 1 — 生成 keyframe 与 image pairs

```bash
python scripts/generate_pairs.py -o outputs/rig_sfm
```

常用参数：

- `--max-keyframes N`：限制 keyframe 数量（快速试跑）
- `--min-translation-m`、`--min-yaw-rad`：keyframe 筛选（见 [src/rig_pairs/README.md](src/rig_pairs/README.md)）

产物：`outputs/rig_sfm/keyframes.json`、`pairs-all.txt` 等。

### 步骤 2 — 特征、匹配、三角化、全局 SfM

```bash
python scripts/run_feature_pairs.py -d outputs/rig_sfm
```

可选：

- `--overwrite-features`：强制重提特征
- `--no-triangulation`：只提特征 + 匹配
- `--copy-images`：拷贝 PNG 而非符号链接到 `images/`

成功时 `outputs/rig_sfm/sfm/` 下有 COLMAP 二进制模型；`triangulation_report.json` 中有三角化统计（典型：数千 3D 点、亚像素级重投影误差量级）。

### 步骤 3 — 单帧 query 定位

在 `test_data` 里选一个存在的 timestamp（µs），例如：

```bash
python scripts/run_query.py \
  -d outputs/rig_sfm \
  --ts 1400075229218675 \
  --camera left \
  --retrieval-top-k 5
```

流程：**NetVLAD 检索 top-k 地图帧 → 仅对这 k 对做 LightGlue → PnP**。依赖步骤 2 生成的 **`global_features.h5`**。

---

## 流水线与模块说明

| 阶段 | 模块 | 说明 |
|------|------|------|
| Pairs | [src/rig_pairs/README.md](src/rig_pairs/README.md) | INS 驱动的 keyframe + 时空 / 基线 pairs |
| 建图 | [src/feature_pairs/README.md](src/feature_pairs/README.md) | hloc DISK/LightGlue + 已知位姿三角化 |
| 定位 | [src/query/README.md](src/query/README.md) | 层次化定位（检索 + 局部匹配 + PnP） |

`source scripts/env.sh` 会设置：

- `PYTHONPATH=v2.0/src`
- `HLOC_ROOT=v2.0/third_party/Hierarchical-Localization`
- `TORCH_HOME=v2.0/model_cache/torch_hub`
- `KMP_DUPLICATE_LIB_OK=TRUE`

---

## 常见问题

| 现象 | 处理 |
|------|------|
| `apply_schubki_hloc_patches.sh` 报 hloc 找不到 | 先 `git submodule update --init`；或忽略 shell 里**旧的** `HLOC_ROOT`，重新 `source scripts/env.sh` |
| `pip` 装到 Python 3.9 user site | 激活 v2.0 venv，用 **`python -m pip install -r requirements.txt`** |
| `lightglue` 依赖冲突 | 不要在外层 `requirements.txt` 再 pin 一份 LightGlue；只通过 hloc submodule 安装 |
| macOS query 段 SIGSEGV / OpenMP | 确认 `KMP_DUPLICATE_LIB_OK=TRUE`；NetVLAD 首次下载完成后重跑 §4 |
| 三角化 0 点 | 确认已执行 `apply_schubki_hloc_patches.sh`（含 `calibrated_two_view_geometry`） |
| 无 GPU | hloc 补丁支持 **MPS（Apple GPU）** 或 CPU，速度较慢 |
| **Query 很慢** | 见 [src/query/README.md](src/query/README.md)：``--fast`` / ``QueryConfig.fast()``（`NN-mutual` + 小 k）、确保建图已有 ``global_features.h5``、重复 query 勿加 ``--force`` |
| `git status` 显示 `third_party/Hierarchical-Localization` modified | **正常**：补丁打在 submodule 工作区里，**不要** `git add third_party/…` 提交。可提交的只有父仓库里的 **`patches/hloc/`**；clone 后跑 `apply_schubki_hloc_patches.sh`。`.gitmodules` 已对 hloc 设置 `ignore = dirty` |

### Submodule 与补丁（不要提交 third_party 内改动）

| 目录 | 父仓库记录什么 | 本地额外步骤 |
|------|----------------|--------------|
| `third_party/robotcar-dataset-sdk` | 固定 **upstream 的 commit SHA** | 一般无需改 |
| `third_party/Hierarchical-Localization` | 同上（cvg 官方 commit） | **`bash scripts/apply_schubki_hloc_patches.sh`** 覆盖 MPS / 三角化等 |

SchubKI 的 hloc 差异保存在 **`patches/hloc/`**，不要进 submodule 里 commit，也不要把「打过补丁的 submodule 指针」推上 GitHub（除非你自己 fork 了 hloc 并改 submodule URL）。

恢复 submodule 到官方干净树（会清掉本地补丁，需再 apply）：

```bash
git submodule update --init third_party/Hierarchical-Localization
bash scripts/apply_schubki_hloc_patches.sh
```

---

## 当前状态

**pairs + feature_pairs + query** 已在 `test_data` 默认序列上打通；可选 **BA + INS prior** 待接。

若从旧 monorepo 迁移：父目录的 `hloc_sdk`、`robot_car_sdk` 已不再需要；依赖 **`v2.0/src`** + **`third_party/`** 即可。
