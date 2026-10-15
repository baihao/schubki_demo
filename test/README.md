# v2.0 测试 Notebook

| 文件 | 说明 |
|------|------|
| `pipeline_one_minute.ipynb` | 20 s 片段 → rig_pairs → feature_pairs → 地图 3D → query（LightGlue） |
| `query_fast_demo.ipynb` | 2 min 建图（`max_temporal_gap=3`）+ **Fast query**（`QueryConfig.fast()`）→ `outputs/segment_2min_gap3_rig/` |

**Kernel：** `SchubKI v2.0`（kernelspec 名 `schubki-v2`）

**前置：**

```bash
cd v2.0
source scripts/env.sh
bash scripts/install_jupyter_kernel.sh   # 首次或 kernel 选不到时重跑
```

**Cursor 里选不到 kernel 时：**

1. 命令面板 → **Notebook: Select Notebook Kernel**
2. **Python Environments** → **Enter interpreter path…** → 选  
   `SchubKI/v2.0/.venv-schubki-v2/bin/python`  
   （仓库根目录已配置 `.vscode/settings.json` 默认解释器，也可在列表里找 **SchubKI v2.0** / `.venv-schubki-v2`）
3. 或 **Jupyter Kernel…** → **SchubKI v2.0**
4. 仍无列表：重载窗口（Developer: Reload Window），确认 `~/Library/Jupyter/kernels/schubki-v2/kernel.json` 存在

产物默认写入 `v2.0/test/outputs/one_minute_rig/`（约 60 keyframes × 2 目，特征/匹配较耗时）。
