# SchubKI hloc patches

Applied onto `third_party/Hierarchical-Localization` after submodule checkout:

- **`utils/device.py`** — CUDA → MPS → CPU inference
- **`extract_features.py` / `match_features.py` / `pairs_from_retrieval.py`** — use `inference_device`
- **`triangulation.py`** — `calibrated_two_view_geometry` for INS-fixed-pose triangulation (pycolmap 4.x)

```bash
bash scripts/apply_schubki_hloc_patches.sh
```
