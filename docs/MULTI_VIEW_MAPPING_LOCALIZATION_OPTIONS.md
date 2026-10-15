# 双目 / 多视图建图与定位：方案探索

本文在 **SchubKI 导论与竖切结论**（v1.0 窗口地标、v2.0 INS 已知位姿三角化、HF-Net / hloc 分层定位阅读）之上，系统梳理：**用双目或多张图像做建图与定位** 时有哪些可行路线、各自依赖哪些技术、以及代表性文献。**不讨论具体代码实现。**

---

## 1. 问题界定

### 1.1 要解决什么

给定车载（或机载）**同步/近似同步** 的图像流，以及可选的 **外参、内参、INS/GNSS**，完成两类任务：

| 任务 | 输出 | 评价要点 |
|------|------|----------|
| **建图（Mapping）** | 稀疏/半稠密 3D 点（及可选描述子）、注册相机位姿、度量尺度 | 尺度是否 metric、点数是否可控、重投影误差、与 INS 一致性 |
| **定位（Localization）** | 查询图相对地图的 **6-DoF 位姿** | 召回率、位置/朝向误差、昼夜/季节鲁棒性 |

本文聚焦 **视觉为主** 的路线；INS 作为 **位姿先验 / 尺度锚定 / 时间外参**，而非纯惯导导航。

### 1.2 「双目」与「多图像」的几何含义

| 观测配置 | 视差来源 | 尺度 | 典型风险 |
|----------|----------|------|----------|
| **同时刻双目（立体）** | 固定基线 \(b\) | 由 \(b\) 直接 metric | 基线短 → 远点不确定；标定与同步误差敏感 |
| **时间多视图（单目沿轨迹）** | 运动产生的基线 | 纯视觉时尺度任意；有 INS/立体可锚定 | **前向运动** 视差弱（degeneracy） |
| **Rig 多相机（≥2 目跨时刻）** | 立体 × 时间 × 跨目 | 立体/INS 给尺度 | pairs 组合爆炸；跨目误匹配 |
| **已知位姿多视图** | 任意已注册相机间的射线交会 | 位姿源决定尺度 | 位姿噪声 / 时钟偏移直接污染三角化 |

SchubKI 场景（Oxford RobotCar 类、前向行车、宽基线立体 + INS）属于：**立体提供短基线 metric，时间链提供主 parallax，INS 提供全局位姿与尺度一致性**。

### 1.3 导论已得到的结论（约束后续方案）

1. **增量 SfM** 在前向行车短序列上易缺好 initial pair，尺度任意，与 INS 无关。
2. **自研 track + 立体 lift（v1.0）** 可控、metric，但产物非标准 SfM 模型，定位需适配层。
3. **已知位姿三角化（v2.0）** 可复用学习特征匹配后再三角化，产出标准稀疏地图，尺度来自 INS/立体。
4. **时钟偏移** 不可忽视（v1.0 测得约 **12.18 s**）；未补偿的 INS→相机位姿会系统性毁掉三角化与定位。
5. **分层定位**（检索 → 局部匹配 → PnP）仍是道路长期定位的工程主线；匹配器可从 NN 升级到 LightGlue；检索可从 NetVLAD 升级到 MegaLoc 等。
6. 大规模建图应使用 **稀疏 pairs（立体 + 时间邻域）**，而非 exhaustive matching。

---

## 2. 技术栈总览（建图 × 定位）

```text
                    ┌─────────────────────────────────────────┐
  传感器            │  相机×N  ·  立体外参  ·  INS/GNSS        │
                    └──────────────────┬──────────────────────┘
                                       ▼
                    ┌─────────────────────────────────────────┐
  预处理            │  去畸变 · 时间同步/偏移 · 关键帧 · pairs │
                    └──────────────────┬──────────────────────┘
                                       ▼
          ┌────────────────────────────┴────────────────────────────┐
          ▼                                                         ▼
   【建图】特征→匹配→三角化/(BA)                              【定位】VPR→匹配→PnP
          │                                                         │
          ▼                                                         ▼
   稀疏地图 (points3D + 描述子)  ◄──────────────────────────►  查询位姿
```

下文方案均落在上述骨架的不同「位姿来源 / 地图表征 / 匹配方式」选择上。

---

## 3. 方案目录（可能路线）

| ID | 方案简称 | 建图核心 | 定位核心 | 与 SchubKI 关系 |
|----|----------|----------|----------|-----------------|
| **A** | 增量多视图 SfM | 增量 SfM（如 COLMAP） | 分层定位 | demo 基线；前向弱 |
| **B** | 已知位姿三角化（Rig+INS） | 固定位姿三角化 | 同 A，metric 地图 | **v2.0 主推** |
| **C** | 立体深度 / track lift | 立体三角化 + 跟踪 | 自研或导出描述子后 PnP | **v1.0 已验** |
| **D** | 分层定位（地图已有） | （依赖 A/B/C） | HF-Net / 分层流水线 | 定位侧标准答案 |
| **E** | 正式多相机 Rig BA | Rig 约束 BA | 同 D | B 的进阶 |
| **F** | 检索+匹配升级 | 同 A/B | MegaLoc + LightGlue | 正交增强 |
| **G** | 基础模型几何（DUSt3R/MASt3R） | 对偶视图点图 / 局部重建 | map-free 或局部配准 | 探索线 |
| **H** | 神经辐射/高斯地图 | 3DGS / NeRF 类 | 渲染对齐或特征场 | 远期 |
| **I** | 立体/多目 VO–SLAM | 在线位姿+局部图 | 重定位 / 闭环 | 在线互补 |

---

## 4. 方案详述与文献

### 4.1 方案 A — 增量多视图 SfM + 事后定位

**思路：** 仅用图像（可加立体 pairs），用增量重建估计位姿与 3D 点；定位时在该模型上做 2D–3D 匹配与 PnP。

**关键技术：**

1. **增量 Structure-from-Motion**  
   选初始像对 → 两视图几何 → 三角化 → 注册新图 → 局部/全局 BA，迭代扩展。  
   - [Schönberger & Frahm, *Structure-from-Motion Revisited*, CVPR 2016](https://openaccess.thecvf.com/content_cvpr_2016/papers/Schonberger_Structure-from-Motion_Revisited_CVPR_2016_paper.pdf)  
   - [Schönberger et al., *Pixelwise View Selection for Unstructured Multi-View Stereo*, ECCV 2016](https://www.cvg.ethz.ch/research/colmap/Schonberger_ECCV2016.pdf)（MVS 扩展，建图稠密时相关）

2. **稀疏特征与匹配**  
   SIFT / DISK / SuperPoint + 最近邻或学习匹配；几何验证用本质/基础矩阵 RANSAC。  
   - [Lowe, *Distinctive Image Features from Scale-Invariant Keypoints*, IJCV 2004](https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf)  
   - [Tyszkiewicz et al., *DISK*, NeurIPS 2020](https://arxiv.org/abs/2006.13566)  
   - [DeTone et al., *SuperPoint*, CVPRW 2018](https://arxiv.org/abs/1712.07629)

3. **分层定位（地图建成后）**  
   见方案 D。

**优点：** 工具链成熟；不依赖 INS。  
**缺点：** 尺度任意；前向行车初始对与角度退化；与 INS 轨迹对齐需额外 Sim(3)。  
**适用：** 无可靠位姿先验的离线建图；或作对照基线。

---

### 4.2 方案 B — 已知位姿三角化（双目 Rig + INS）★ 主推

**思路：** 用 INS（经时钟偏移校正）+ 立体外参写出每帧每相机的 **已知相机位姿**，再仅做 **多视图三角化**（及可选 BA），跳过增量 mapper 的位姿估计。

**关键技术：**

1. **已知相机位姿下的三角化**  
   多射线 DLT / 中点法 + 重投影优化。几何验证可用 **已知相对位姿下的极线误差**（比盲目 RANSAC 更稳）。  
   - Hartley & Zisserman, *Multiple View Geometry in Computer Vision*, 2nd ed., Cambridge University Press, 2004（三角化、极线几何）  
   - [Schönberger & Frahm, SfM Revisited, CVPR 2016](https://openaccess.thecvf.com/content_cvpr_2016/papers/Schonberger_Structure-from-Motion_Revisited_CVPR_2016_paper.pdf)（COLMAP 中三角化 / 已知位姿重建管线）

2. **多传感器 Rig / 车体坐标系**  
   同一 timestamp 下多相机共享车体位姿，传感器相对外参固定（标定）。BA 时可只优化车体位姿、锁定外参。  
   - [Kaess et al., *iSAM2*, IJRR 2012](https://www.cs.cmu.edu/~kaess/pub/Kaess12ijrr.pdf)（因子图多传感器融合脉络）  
   - [Fleps et al., *Optimization Based Multi-Scale Calibration*, ICRA 2011](https://doi.org/10.1109/ICRA.2011.5980298) / Kalibr 相关标定方法学

3. **INS–视觉时间外参**  
   图像时间戳与 INS 时间基不一致时，需估计常数或缓变偏移。导论结论：**未补偿偏移会使「已知位姿」失效**。  
   - [Kelly & Sukhatme, *Visual-Inertial Sensor Fusion…*, IJRR 2011](https://doi.org/10.1177/0278364910386986)  
   - [Furgale et al., *Unified Temporal and Spatial Calibration…*, IROS 2013](https://doi.org/10.1109/IROS.2013.6696514)（时空联合标定）

4. **Pairs 图设计（立体 + 时间 + 可选跨目）**  
   - 同 ts 左右：固定基线，锚定尺度与短程深度；  
   - 同目时间邻域：主 parallax 来源；  
   - 跨目短时：增强连通，控制边数。  
   避免 exhaustive，降低误匹配与计算量（与大规模 SfM 稀疏图思想一致，见 SfM Revisited）。

5. **可选 Bundle Adjustment + 位姿先验**  
   固定或弱扰动车体位姿，优化 3D 点；或对 INS 位姿加高斯先验。  
   - [Triggs et al., *Bundle Adjustment — A Modern Synthesis*, 1999](https://link.springer.com/chapter/10.1007/3-540-44480-7_21)

**优点：** metric；可与标准分层定位接口兼容；避开增量初始化失败。  
**缺点：** 依赖标定、同步与 INS 质量；纯前向仍需最小三角化角等过滤。  
**适用：** SchubKI / RobotCar 类 **有 INS + 立体标定** 的离线建图。

**直接相关工作：**

- [Sarlin et al., *From Coarse to Fine…* (HF-Net), CVPR 2019](https://arxiv.org/abs/1812.03506) — 分层定位框架；大规模定位中常用「先有地图位姿/模型，再三角化或匹配」的工程拆分  
- [Maddern et al., *Oxford RobotCar Dataset*, IJRR 2017](https://doi.org/10.1177/0278364916679498) — 本项目数据与多会话定位语境

---

### 4.3 方案 C — 立体几何 Lift + 多帧 Track（v1.0 路线）

**思路：** 在特征匹配上建 **跨帧 track**，对同时具有左右观测的 track 用 **立体三角化** 得 metric 3D，再经重投影 / 长度 / 视差过滤。

**关键技术：**

1. **立体视觉三角化**  
   视差 \(d\) 与深度 \(Z = fb/d\)；需去畸变与校正或显式使用投影矩阵。  
   - Hartley & Zisserman, *Multiple View Geometry*, 2004  
   - [Hirschmüller, *Stereo Processing by Semiglobal Matching…*, CVPR 2005 / PAMI 2008](https://doi.org/10.1109/TPAMI.2007.1166)（稠密立体脉络；稀疏立体三角化同属立体几何）

2. **多视图 Track 管理**  
   将立体匹配边与时间匹配边连成 track；最小长度、最大重投影误差剪枝（SfM track 思想，见 SfM Revisited）。

3. **匹配质量门控**  
   描述子距离、学习匹配器分数、mutual check、[Lowe ratio test](https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf)。

**优点：** 点少可控；尺度明确；实现路径短。  
**缺点：** 地图格式往往自研；与通用定位工具耦合弱；跨目/长时间一致性不如完整 SfM BA。  
**适用：** 快速验证「立体 + 时间」能否出稳地标；与方案 B 做 ablation。

---

### 4.4 方案 D — 分层视觉定位（地图已由 A/B/C 给出）

**思路：** 不重新估计整幅地图几何；查询时 **粗检索地点 → 细匹配局部特征 → 2D–3D → PnP(+RANSAC)**。

**关键技术：**

1. **视觉地点识别（VPR / 图像检索）**  
   全局描述子 + 近邻搜索，召回同地点候选图。  
   - [Arandjelović et al., *NetVLAD*, CVPR 2016](https://arxiv.org/abs/1511.07247)  
   - [Berton et al., *Rethinking Visual Geo-localization…* (CosPlace), CVPR 2022](https://arxiv.org/abs/2204.02287)  
   - [Berton & Masone, *MegaLoc*, CVPRW 2025](https://arxiv.org/abs/2502.17237)  
   - [Keetha et al., *AnyLoc*, IEEE RA-L 2023](https://arxiv.org/abs/2308.00688)

2. **局部特征与学习匹配**  
   - [DeTone et al., *SuperPoint*, CVPRW 2018](https://arxiv.org/abs/1712.07629)  
   - [Sarlin et al., *SuperGlue*, CVPR 2020](https://arxiv.org/abs/1911.11763)  
   - [Lindenberger et al., *LightGlue*, ICCV 2023](https://arxiv.org/abs/2306.13643) — 自适应深度/剪枝  
   - [Sarlin et al., HF-Net, CVPR 2019](https://arxiv.org/abs/1812.03506) — 局部侧原为 **NN + ratio**；学习匹配器权重通常与特定特征（如 SuperPoint）绑定训练

3. **绝对位姿：PnP + RANSAC**  
   - [Lepetit et al., *EPnP*, IJCV 2009](https://doi.org/10.1007/s11263-008-0152-6)  
   - [Fischler & Bolles, *RANSAC*, CACM 1981](https://doi.org/10.1145/358669.358692)

4. **层次化流水线**  
   - [Sarlin et al., HF-Net, CVPR 2019](https://arxiv.org/abs/1812.03506)  
   - [Sattler et al., *Benchmarking 6DOF Outdoor Visual Localization…*, CVPR 2018](https://arxiv.org/abs/1707.08370)（长期视觉定位评测语境）

**优点：** 成熟、可换检索/匹配模块；与地图格式解耦（只要有 2D–3D 描述子）。  
**缺点：** 地图质量与覆盖决定上限；外观变化大时依赖 VPR 与匹配鲁棒性。  
**适用：** SchubKI **查询阶段** 的默认答案；建图侧选 A/B/C 任一产物即可接入。  

**流程拆解：** 见下文 **§5**（从 SfM 模型反推新图像位姿）。

---

### 4.5 方案 E — 正式多相机 Rig 建模与联合优化

**思路：** 在方案 B 之上，用显式 **刚体多相机约束**（而非每图独立位姿）表达外参，在 BA 中固定传感器相对外参，仅优化车体位姿与 3D 点。可扩展第三相机作辅助观测，**不宜与主立体宽基线混为同一深度对**（不同焦距/基线几何不一致）。

**关键技术：**

1. **刚体多相机外参约束** — 减少自由度、抑制单目漂移。  
2. **多相机 / 时空标定**  
   - [Furgale et al., *Unified Temporal and Spatial Calibration…*, IROS 2013](https://doi.org/10.1109/IROS.2013.6696514)  
   - [Rehder et al., *Extending Kalibr…*, IROS 2016](https://doi.org/10.1109/IROS.2016.7759050)  
3. **可选跨目 pairs** — 视差轴与运动轴耦合时增加图连通性。

**优点：** 几何一致、BA 更干净。  
**缺点：** 实现与调试成本高于「每图独立位姿」简化版。  
**适用：** 多相机正式产品化、需要严格 rig 约束时。

---

### 4.6 方案 F — 检索与匹配的正交升级（不改地图几何）

**思路：** 保持建图几何（B/C）不变，仅替换定位前端：

- 全局：NetVLAD / CosPlace → **MegaLoc** / AnyLoc；  
- 局部匹配：NN → **LightGlue**（应搭配其训练所对应的特征，如 SuperPoint / DISK）。

**关键技术：** 同方案 D 中 [MegaLoc](https://arxiv.org/abs/2502.17237)、[AnyLoc](https://arxiv.org/abs/2308.00688)、[LightGlue](https://arxiv.org/abs/2306.13643)。  
**优点：** 性价比高；MegaLoc 等工作表明在既有定位管线上 **只换检索** 即可抬升结果。  
**缺点：** 不解决建图尺度/退化问题。  
**适用：** 任意阶段可并行插入的增强项。

---

### 4.7 方案 G — 基础模型几何（DUSt3R / MASt3R 等）

**思路：** 用大规模预训练的 **成对点图 / 局部特征几何网络**，在少先验甚至无先验下估计相对几何或局部点云，再用于 map-free 定位或辅助建图。

**关键技术：**

1. **DUSt3R** — 双视图点图回归，弱化传统标定/位姿管线。  
   - [Wang et al., *DUSt3R*, CVPR 2024](https://arxiv.org/abs/2312.14132)

2. **MASt3R** — 增强局部特征与匹配，并讨论视觉定位设定。  
   - [Leroy et al., *Grounding Image Matching in 3D with MASt3R*, ECCV 2024](https://arxiv.org/abs/2406.09756)

3. **多视图扩展**  
   - [Cabon et al., *MUSt3R: Multi-view Network for Stereo 3D Reconstruction*, CVPR 2025](https://arxiv.org/abs/2503.01661)

**优点：** 大基线、难匹配、少图场景潜力大；可作「非经典 SfM」对照。  
**缺点：** 城市长距离、度量一致性、与 INS 融合仍不成熟。  
**适用：** 探索难例、map-free 分支；**不建议** 作为第一版主路径。

---

### 4.8 方案 H — 神经地图（NeRF / 3D Gaussian Splatting）定位

**思路：** 地图不再是稀疏点 + 描述子，而是可微渲染场或高斯椭球；定位通过 **渲染–实测对齐** 或神经特征场匹配求解位姿。

**关键技术与代表工作：**

- [Kerbl et al., *3D Gaussian Splatting*, SIGGRAPH 2023](https://arxiv.org/abs/2308.04079)  
- [Zhai et al., *SplatLoc*, arXiv:2409.14067](https://arxiv.org/abs/2409.14067)  
- [GSplatLoc, arXiv:2409.16502](https://arxiv.org/abs/2409.16502)  
- [LoGS, arXiv:2410.11505](https://arxiv.org/abs/2410.11505)  
- [GSLoc, arXiv:2410.06165](https://arxiv.org/abs/2410.06165)

**优点：** 新表征、纹理丰富场景位姿细化强。  
**缺点：** 车载长距离存储/更新/动态物体；与经典分层点图评测链不完全同构。  
**适用：** 远期调研；非当前 RobotCar 竖切重点。

---

### 4.9 方案 I — 立体 / 多目视觉里程计与 SLAM（在线）

**思路：** 在线估计相对运动与局部地图，可选闭环；定位可为子地图重定位。立体 VO 可直接 metric。

**关键技术：**

1. **立体视觉里程计** — 特征法或半直接法。  
   - [Geiger et al., *StereoScan / LIBVISO 脉络*, IV 2011 等](https://doi.org/10.1109/IVS.2011.5940400)
2. **ORB-SLAM2/3 立体/多目模式**  
   - [Mur-Artal & Tardós, *ORB-SLAM2*, IEEE TRO 2017](https://arxiv.org/abs/1610.06475)  
   - [Campos et al., *ORB-SLAM3*, IEEE TRO 2021](https://arxiv.org/abs/2007.11898)
3. **VIO** — 视觉–惯性紧耦合（若强化 INS 使用）。  
   - [Qin et al., *VINS-Mono*, IEEE TRO 2018](https://arxiv.org/abs/1708.03852)  
   - [Geneva et al., *OpenVINS*, IROS 2020](https://arxiv.org/abs/2003.05085)

**优点：** 在线、低延迟；立体自然 metric。  
**缺点：** 长期全局一致性与「离线测绘地图 + 用户端定位」产品形态不同；前向高速场景仍难。  
**适用：** 与离线方案 B+D **互补**（局部跟踪 + 全局重定位），而非替代。

---

## 5. 从 SfM 模型反推新图像位姿（定位详解）

给定一张**从未参与建图**的查询图 \(I_q\)，以及一份已建成的 **SfM / 稀疏地图**，目标是估计查询相机在**地图世界坐标系**下的绝对位姿（通常写成 `cam_from_world`：世界点 → 相机坐标的刚体变换）。这与 notebook 里用本质矩阵估**两帧相对运动**不同：这里求的是相对**整张地图**的绝对位姿，尺度与坐标原点由建图阶段决定（若建图用了 INS/立体，则为 metric）。

核心范式见 [HF-Net / 分层定位](https://arxiv.org/abs/1812.03506)：先粗后细，把「在哪一带」和「精确 6-DoF」拆开。

### 5.1 SfM 模型里已经有什么

一份可用的定位地图至少包含：

| 要素 | 含义 | 定位时如何用 |
|------|------|--------------|
| **注册图像（database / reference images）** | 建图时用过的、已有位姿的图 | 作为「外观邻居」与查询做 2D–2D 匹配 |
| **相机内参** | 每张库图（及查询）的 \(K\)、畸变或去畸变后的针孔模型 | PnP 需要查询内参；投影几何必须与特征坐标一致 |
| **3D 点 \(\mathbf{X}_j\)** | 三角化得到的世界坐标点 | PnP 的 3D 端 |
| **2D–3D 观测（track）** | 哪些库图关键点观测了哪个 \(\mathbf{X}_j\) | 把「查询↔库图」的 2D–2D 匹配**提升**为查询↔地图的 2D–3D |
| **（可选）局部描述子** | 与关键点绑定的特征（SuperPoint/DISK/…） | 匹配查询特征与库图特征 |
| **（可选）全局描述子** | 每张库图一个向量 | 图像检索，缩小候选集 |

没有「查询像素直接连到 3D」的魔法：中间桥梁始终是 **库图上已经三角化过的特征点**。

### 5.2 总流程

```text
新图像 I_q
    │
    ├─(1) 提取全局描述子 ──► 在库图中检索 top-k 候选（VPR）
    │
    ├─(2) 提取局部关键点 + 描述子（与建图同一套特征）
    │
    ├─(3) 将 I_q 与每个候选库图做局部匹配（NN / LightGlue / …）
    │         └─► 得到 2D_q ↔ 2D_db
    │
    ├─(4) 借助 SfM track：若 2D_db 对应某个 3D 点 X
    │         └─► 提升为 2D_q ↔ X   （2D–3D 对应）
    │
    ├─(5) （可选）按共视关系聚类候选，避免把远处错误簇混在一起
    │
    └─(6) 用内参 K_q + 2D–3D 做 PnP + RANSAC（可再非线性 refine）
              └─► 输出 cam_from_world（或等价的 R, t）
```

下面逐步说明「为什么」和「在算什么」。

### 5.3 步骤 (1)：全局检索——先缩小「可能在哪」

全库逐张做局部匹配不可行（库图可达海量）。因此先把 \(I_q\) 编码成**全局描述子**，与库图描述子比相似度，取 top-\(k\)（常见 \(k=10\sim50\)）。

- 技术：NetVLAD、CosPlace、MegaLoc、AnyLoc 等（§4.4 / §4.6）。  
- 输出：一组**外观相近**的数据库图像名（尚无精确位姿）。  
- 失败模式：昼夜/季节变化导致检索错段 → 后续匹配再准也定错地方；可用更大 \(k\)、更强 VPR、或多会话库图缓解（见 [Sattler et al., CVPR 2018](https://arxiv.org/abs/1707.08370)）。

检索解决的是 **place recognition**；还不产生 6-DoF。

### 5.4 步骤 (2)：查询局部特征——与建图特征同域

对 \(I_q\) 检测关键点并计算描述子。**必须与建图时同一类特征**（例如库是 DISK，查询也用 DISK；LightGlue 的 `superpoint` 权重不能硬套任意蒸馏描述子）。

- 几何约定：若建图在**去畸变图**上提特征，查询也应去畸变，且 \(K_q\) 与去畸变后的针孔一致。  
- 像素中心惯例：COLMAP 常用「像素中心在半整数」；实现里常对关键点做 \(+0.5\) 对齐（概念上：特征坐标与投影模型一致即可）。

### 5.5 步骤 (3)：查询 ↔ 候选库图的 2D–2D 匹配

对每个检索到的库图 \(I_d\)，在查询关键点与库图关键点之间建立对应：

- 经典：描述子最近邻 + ratio test / mutual check（[Lowe, IJCV 2004](https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf)）。  
- 学习匹配：SuperGlue / [LightGlue](https://arxiv.org/abs/2306.13643) 直接输出匹配与置信度。

得到的是像素对 \((\mathbf{x}_q, \mathbf{x}_d)\)，仍**没有**世界坐标。

### 5.6 步骤 (4)：经 SfM track 提升为 2D–3D——定位的关键一跳

SfM 在建图时已经记录：库图 \(I_d\) 上的某个关键点 \(\mathbf{x}_d\) 是否关联了一个 3D 点 \(\mathbf{X}\)（即该点曾被三角化并保留在 `points3D` 中）。

因此：

\[
\mathbf{x}_q \leftrightarrow \mathbf{x}_d \quad\text{且}\quad \mathbf{x}_d \leftrightarrow \mathbf{X}
\quad\Longrightarrow\quad
\mathbf{x}_q \leftrightarrow \mathbf{X}
\]

对所有候选库图、所有匹配重复此过程，汇总为一组 **2D–3D 对应** \(\{(\mathbf{x}_q^{(i)}, \mathbf{X}^{(i)})\}\)。

注意：

- 库图上**未三角化**的特征（只有 2D、无 3D）对绝对位姿**无直接贡献**，匹配会被丢弃。  
- 同一查询关键点可能通过不同库图连到同一 \(\mathbf{X}\)（应去重）或错误地连到多个 \(\mathbf{X}\)（交给 RANSAC 剔除外点）。  
- 地图点越密、track 越长，能提升的 2D–3D 通常越多，PnP 越稳——这正是建图质量决定定位上限的原因。

### 5.7 步骤 (5)：（可选）共视聚类——避免「混簇」位姿

top-\(k\) 检索可能同时召回**地理上不相邻**的相似外观（重复结构、错误检索）。若把所有 2D–3D 一股脑喂给 PnP，内点可能来自错误地点。

做法（[HF-Net](https://arxiv.org/abs/1812.03506)、hloc 等实现中的常见选项）：在候选库图之间，按「是否观测到共同 3D 点」建图，做**连通分量 / 共视聚类**；对每个簇分别做 PnP，再取内点数最多或重投影最好的位姿。

小场景、检索很准时可关掉；城市长期定位、重复结构多时打开更稳。

### 5.8 步骤 (6)：PnP + RANSAC → 绝对位姿

有了 \(N\) 对 \(\mathbf{x}_q \leftrightarrow \mathbf{X}\) 和查询内参 \(K_q\)，求解：

\[
\min_{R,\,\mathbf{t}}
\sum_{i}
\rho\!\left(
\bigl\|
\pi(K_q,\, R\mathbf{X}^{(i)}+\mathbf{t})
- \mathbf{x}_q^{(i)}
\bigr\|
\right)
\]

其中 \(\pi\) 为透视投影，\(\rho\) 为鲁棒核（RANSAC 内点 / 外点）。

- **最小求解器**：P3P 等（至少 3 对非共线对应）；工程上常用 [EPnP](https://doi.org/10.1007/s11263-008-0152-6) 等在 RANSAC 假设内估计。  
- **RANSAC**：[Fischler & Bolles, 1981](https://doi.org/10.1145/358669.358692) —— 反复采样、数重投影误差小于阈值的内点。  
- **精化**：在内点上做非线性重投影优化（absolute pose refinement），得到最终 \(R,\mathbf{t}\)。

输出含义：

- \(\mathbf{X}_{\text{cam}} = R\,\mathbf{X}_{\text{world}} + \mathbf{t}\)（即 `cam_from_world`）；  
- 相机中心在世界系中为 \(\mathbf{C} = -R^\top\mathbf{t}\)。  

若地图是 metric（方案 B），则 \(\mathbf{C}\) 的单位为米；若地图来自任意尺度增量 SfM（方案 A），则位姿与地图同尺度，需再与 INS 做 Sim(3) 才有物理米制。

**成功判据（实践）：** 内点数足够（常需数十以上，视场景而定）、重投影误差小、位姿与邻帧/先验不突变。失败则增大检索 \(k\)、换更强匹配、检查内参与是否去畸变一致。

### 5.9 与「本质矩阵相对位姿」的对比

| | 本质矩阵 / `recoverPose`（如 notebook 双帧 VO） | SfM 绝对定位（本节） |
|--|-----------------------------------------------|----------------------|
| 输入 | 两帧之间的 2D–2D + \(K\) | 查询 2D–3D（经地图）+ \(K_q\) |
| 输出 | 相对 \(R\)、**单位**平移方向 \(t\) | 相对**世界/地图**的 \(R,\mathbf{t}\)（可有尺度） |
| 尺度 | 单目无法定尺度 | 继承地图尺度 |
| 角色 | 里程计、短时相对运动 | 全局重定位 / 度量定位 |

二者可互补：VO 提供帧间平滑；丢失跟踪或需要全局坐标时，用本节方法相对 SfM 地图重定位。

### 5.10 前置条件清单（否则位姿会 silently 错）

1. 查询 **内参正确**，且与特征所在图像几何一致（畸变是否已去除）。  
2. 查询特征与库图特征 **同模型、同预处理**（分辨率缩放时 \(K\) 要同比缩放，如 notebook 中 `scale=0.5`）。  
3. 地图中有足够 **已三角化** 点，且查询地点被库图覆盖。  
4. 检索未系统性跑偏；必要时开共视聚类。  
5. PnP RANSAC 阈值与图像分辨率匹配（像素误差阈值过大/过小都会伤内点质量）。

### 5.11 本节主要文献

- [Sarlin et al., HF-Net, CVPR 2019](https://arxiv.org/abs/1812.03506) — 粗到细分层定位  
- [Sattler et al., CVPR 2018](https://arxiv.org/abs/1707.08370) — 户外长期定位评测与挑战  
- [Lepetit et al., EPnP, IJCV 2009](https://doi.org/10.1007/s11263-008-0152-6)；[Fischler & Bolles, RANSAC, 1981](https://doi.org/10.1145/358669.358692)  
- [Lindenberger et al., LightGlue, ICCV 2023](https://arxiv.org/abs/2306.13643)；[Berton & Masone, MegaLoc, 2025](https://arxiv.org/abs/2502.17237) — 匹配与检索升级  
- Hartley & Zisserman, *Multiple View Geometry* — PnP / 投影几何基础  

---

## 6. 方案对比（面向 SchubKI）

| 维度 | A 增量 SfM | B 已知位姿三角化 | C 立体 Track | D 分层定位 | G 基础模型 | H 3DGS | I VO/SLAM |
|------|------------|------------------|--------------|------------|------------|--------|-----------|
| 尺度 | 任意 | metric（INS/立体） | metric | 继承地图 | 视训练/对齐 | 视建图 | 立体可 metric |
| 前向行车稳健性 | 弱 | 中–强（依赖 pairs/过滤） | 中 | 取决于地图 | 潜力大 | 未验证长距 | 中 |
| 与分层定位接口 | 原生 | 原生 | 需适配 | 原生 | 部分设定 | 另套 | 另套 |
| 对 INS 依赖 | 无 | 强（含时间偏移） | 弱–中 | 无 | 无 | 无 | 可选 VIO |
| 实现复杂度 | 低（现成） | 中 | 中 | 低 | 中–高 | 高 | 中 |
| 导论推荐优先级 | 对照基线 | **P0 建图** | 已完成对照 | **P0 定位** | 探索 | 远期 | 可选在线 |

---

## 7. 推荐组合（由导论收敛）

```text
建图：方案 B（INS 参考位姿 + 立体/时间 pairs + 三角化 [+ 可选 BA/Rig]）
      ↑ 可用 C 的过滤思想（track 长、重投影、三角化角）作后处理

定位：方案 D + 方案 F
      （MegaLoc/强 VPR 可选）+（DISK/SuperPoint + LightGlue）+ PnP

对照：方案 A（同数据上的增量 SfM）
探索：方案 G（难例）；方案 E（Rig 正式化）；方案 I（在线）；方案 H（远期）
```

**必须贯穿所有「已知位姿」方案的前置条件：**

1. 相机内参 + 立体/多目外参可靠；图像与特征在同一几何（建议去畸变后的针孔模型）。  
2. **INS–图像时间偏移** 已标定。  
3. pairs 覆盖 **立体边 + 时间边**，控制跨目边数量。  
4. 三角化后按 **最小射线夹角、track 长度、重投影误差** 过滤，抑制前向退化远点。

---

## 8. 参考文献（按主题，均为已发表文献）

### 8.1 多视图几何与 SfM

1. R. Hartley, A. Zisserman. *Multiple View Geometry in Computer Vision*. 2nd ed., Cambridge University Press, 2004.  
2. [J. L. Schönberger, J.-M. Frahm. Structure-from-Motion Revisited. CVPR, 2016.](https://openaccess.thecvf.com/content_cvpr_2016/papers/Schonberger_Structure-from-Motion_Revisited_CVPR_2016_paper.pdf)  
3. [B. Triggs et al. Bundle Adjustment — A Modern Synthesis. 1999.](https://link.springer.com/chapter/10.1007/3-540-44480-7_21)

### 8.2 分层定位与长期视觉定位

4. [P.-E. Sarlin et al. From Coarse to Fine: Robust Hierarchical Localization at Large Scale. CVPR, 2019.](https://arxiv.org/abs/1812.03506)  
5. [T. Sattler et al. Benchmarking 6DOF Outdoor Visual Localization in Changing Conditions. CVPR, 2018.](https://arxiv.org/abs/1707.08370)  
6. [W. Maddern et al. 1 Year, 1000 km: The Oxford RobotCar Dataset. IJRR, 2017.](https://doi.org/10.1177/0278364916679498)

### 8.3 特征、匹配与检索

7. [D. G. Lowe. Distinctive Image Features from Scale-Invariant Keypoints. IJCV, 2004.](https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf)  
8. [D. DeTone et al. SuperPoint. CVPR Workshops, 2018.](https://arxiv.org/abs/1712.07629)  
9. [P.-E. Sarlin et al. SuperGlue. CVPR, 2020.](https://arxiv.org/abs/1911.11763)  
10. [P. Lindenberger et al. LightGlue. ICCV, 2023.](https://arxiv.org/abs/2306.13643)  
11. [M. Tyszkiewicz et al. DISK. NeurIPS, 2020.](https://arxiv.org/abs/2006.13566)  
12. [R. Arandjelović et al. NetVLAD. CVPR, 2016.](https://arxiv.org/abs/1511.07247)  
13. [G. Berton et al. CosPlace. CVPR, 2022.](https://arxiv.org/abs/2204.02287)  
14. [G. Berton, C. Masone. MegaLoc. CVPRW, 2025.](https://arxiv.org/abs/2502.17237)  
15. [N. Keetha et al. AnyLoc. IEEE RA-L, 2023.](https://arxiv.org/abs/2308.00688)

### 8.4 位姿求解与标定

16. [V. Lepetit et al. EPnP. IJCV, 2009.](https://doi.org/10.1007/s11263-008-0152-6)  
17. [M. A. Fischler, R. C. Bolles. RANSAC. CACM, 1981.](https://doi.org/10.1145/358669.358692)  
18. [P. Furgale et al. Unified Temporal and Spatial Calibration… IROS, 2013.](https://doi.org/10.1109/IROS.2013.6696514)  
19. [J. Kelly, G. S. Sukhatme. Visual-Inertial Sensor Fusion… IJRR, 2011.](https://doi.org/10.1177/0278364910386986)

### 8.5 基础模型几何与神经地图

20. [S. Wang et al. DUSt3R. CVPR, 2024.](https://arxiv.org/abs/2312.14132)  
21. [V. Leroy et al. MASt3R. ECCV, 2024.](https://arxiv.org/abs/2406.09756)  
22. [B. Kerbl et al. 3D Gaussian Splatting. SIGGRAPH, 2023.](https://arxiv.org/abs/2308.04079)  
23. [SplatLoc. arXiv:2409.14067.](https://arxiv.org/abs/2409.14067)  
24. [GSplatLoc. arXiv:2409.16502.](https://arxiv.org/abs/2409.16502)  
25. [LoGS. arXiv:2410.11505.](https://arxiv.org/abs/2410.11505)  
26. [GSLoc. arXiv:2410.06165.](https://arxiv.org/abs/2410.06165)

### 8.6 在线 SLAM / VO

27. [R. Mur-Artal, J. D. Tardós. ORB-SLAM2. IEEE TRO, 2017.](https://arxiv.org/abs/1610.06475)  
28. [C. Campos et al. ORB-SLAM3. IEEE TRO, 2021.](https://arxiv.org/abs/2007.11898)  
29. [T. Qin et al. VINS-Mono. IEEE TRO, 2018.](https://arxiv.org/abs/1708.03852)  
30. [P. Geneva et al. OpenVINS. IROS, 2020.](https://arxiv.org/abs/2003.05085)

### 8.7 车载系统向（扩展阅读）

31. [SF-Loc. arXiv:2412.01500.](https://arxiv.org/abs/2412.01500)  
32. [LiteVLoc. arXiv:2410.04419.](https://arxiv.org/abs/2410.04419)

---

## 9. 小结

- **双目** 解决的是 **短基线 metric 深度与尺度锚定**；**多图像（时间/多相机）** 解决的是 **视差累积、track 与全局一致性**。二者应组合，而不是二选一。  
- 在有 **INS + 立体标定** 的前提下，导论收敛的建图答案是 **已知位姿三角化（方案 B）**，定位答案是 **分层定位（方案 D）**，并用 **更强检索/匹配（方案 F）** 提升上限。  
- **新图像位姿**：不重新跑 SfM，而是检索 → 局部匹配 → 经库图 track 得到 2D–3D → PnP（§5）。  
- 增量 SfM、自研立体 lift、基础模型几何、神经地图、在线 SLAM 分别承担 **对照、已验证备选、探索、远期、在线互补** 角色，避免在主路径上堆叠未验证范式。

---

*文档性质：方案与已发表文献梳理；不含仓库内实现文档链接。*
