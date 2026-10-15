# `rig_pairs` — Keyframe 与 hloc Pairs 生成

基于 **`test_data`** 的 Oxford RobotCar 宽基线立体（left/right 分目录、按 ts 同步），生成 hloc 所需的 **`pairs-*.txt`** 与 **`keyframes.json`**。**不包含** SfM 三角化或定位。

设计依据：[../docs/RIG_KNOWN_POSE_TRIANGULATION.md](../docs/RIG_KNOWN_POSE_TRIANGULATION.md) §5、§9。

---

## 依赖

- **`src/robotcar`**（`pip install -r requirements-robot-car-sdk.txt` + `source scripts/env.sh`）
- **`test_data`** 默认序列 `2014-05-14-13-46-12-*`

---

## 导入

```bash
cd v2.0
source ../.venv-schubki-v2/bin/activate   # 或: source scripts/env.sh
export PYTHONPATH="${PWD}/src:${PYTHONPATH}"
```

---

## API

| 符号 | 说明 |
|------|------|
| `KeyframeConfig` | 数据路径、INS 偏移、按距离/yaw 选 keyframe；或 `start_ts` + `segment_duration_us` + `keyframe_interval_us`（按秒采样） |
| `PairsConfig` | 时间边 gap、最小光心距、跨目 |
| `load_synced_stereo` / `select_keyframes` | 同步与选帧 |
| `build_pair_lists` / `write_pair_files` | 生成并写入 pairs |
| `generate_pairs_for_test_data` | 一键 pipeline |

---

## 示例

```python
from pathlib import Path
from rig_pairs import KeyframeConfig, PairsConfig, generate_pairs_for_test_data

keyframes, lists, paths = generate_pairs_for_test_data(
    output_dir=Path("outputs/rig_sfm"),
    keyframe_config=KeyframeConfig(max_keyframes=30),
    pairs_config=PairsConfig(min_baseline_m=0.75),
)
```

CLI：`python scripts/generate_pairs.py -o outputs/rig_sfm`

---

## 输出

| 文件 | 内容 |
|------|------|
| `keyframes.json` | 选中 keyframe 元数据 |
| `pairs-*.txt` | 立体 / 时间 / 合并列表 |

默认写入 **`outputs/rig_sfm/`**（整条建图流水线工作目录名不变；本 Python 包仅负责 pairs 段）。
