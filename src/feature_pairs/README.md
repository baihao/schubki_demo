# `feature_pairs` — 从 rig pairs 到全局 SfM

在 **`rig_pairs`** 产物（`keyframes.json`、`pairs-all.txt`）之上，用 **`hloc_bridge`** 完成：

1. 对 pairs 涉及的全部图像 **提取 DISK 特征**
2. 按 **`pairs-all.txt`** **匹配**
3. 地图 **NetVLAD** → **`global_features.h5`**（供 query 全局检索）
4. 建 **INS reference** + **三角化验证**（每图 triangulated/keypoints）
5. 输出全局 **`sfm/`** COLMAP 模型 + **`triangulation_report.json`**

前置：已运行 `python scripts/generate_pairs.py -o outputs/rig_sfm`。

---

## 依赖

- `src/robotcar`（标定、INS）
- `src/hloc_bridge` + 可 `import hloc` / `pycolmap` / `torch` 的环境
- `third_party/robotcar-dataset-sdk` 的 `models/`、`extrinsics/`（`bash scripts/setup_robotcar_sdk.sh` 或 submodule）

```bash
cd v2.0
git submodule update --init --recursive
bash scripts/apply_schubki_hloc_patches.sh
source scripts/env.sh
pip install -r requirements.txt
```

---

## CLI

```bash
python scripts/run_feature_pairs.py -d outputs/rig_sfm
python scripts/run_feature_pairs.py -d outputs/rig_sfm --overwrite-features
python scripts/run_feature_pairs.py -d outputs/rig_sfm --no-triangulation  # 仅特征+匹配
```

---

## API

```python
from pathlib import Path
from feature_pairs import FeaturePairsConfig, FeaturePairsPaths, run_feature_pairs_pipeline

result = run_feature_pairs_pipeline(
    FeaturePairsConfig(
        paths=FeaturePairsPaths(rig_sfm_dir=Path("outputs/rig_sfm")),
        overwrite_features=False,
    )
)
print(result.graph)
print(result.validation.mean_reprojection_error if result.validation else "skipped")
```

---

## 目录布局（`rig_sfm_dir`）

| 输入（rig_pairs） | 本模块写入 |
|-------------------|------------|
| `keyframes.json`, `pairs-all.txt` | `images/`（链到 test_data 去畸变源图） |
| | `features.h5`, `matches.h5` |
| | `reference/`（INS 位姿，无点） |
| | `sfm/`（三角化后稀疏地图） |
| | `triangulation_report.json` |

---

## 与文档

实现对应 [RIG_KNOWN_POSE_TRIANGULATION.md](../docs/RIG_KNOWN_POSE_TRIANGULATION.md) §4、§6、§7 步骤 B、§8 步骤 C。
