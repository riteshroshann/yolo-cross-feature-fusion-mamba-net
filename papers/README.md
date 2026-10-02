# Papers

Every work cited in the research dossier (`report/CFFM-Net_Research_Dossier.pdf`), grouped as in the report. Open-access papers are stored here as PDFs; the rest are listed with the reason and a link. `index.csv` holds the same information in machine-readable form, keyed by the citation keys of `report/references.tex`.

148 of 174 cited works are stored as PDFs. To refresh, run `python papers/download_papers.py` (files already present are skipped). arXiv identifiers were confirmed against each PDF's first page.

## Source papers supplied with the topic

- [G17-2025 CVPR-A Hybrid Mamba-Transformer Vision Backbone](00_source_papers/G17-2025%20CVPR-A%20Hybrid%20Mamba-Transformer%20Vision%20Backbone.pdf)
- [G17-2025 Enhancing Small Object Detection Using Transformer](00_source_papers/G17-2025%20Enhancing%20Small%20Object%20Detection%20Using%20Transformer.pdf)
- [G17-2026 A CNN–Mamba hybrid network for efficient small object detection](00_source_papers/G17-2026%20A%20CNN%E2%80%93Mamba%20hybrid%20network%20for%20efficient%20small%20object%20detection.pdf)
- [G17-2026 Cross-Modality Fusion Mamba for Object Detection](00_source_papers/G17-2026%20Cross-Modality%20Fusion%20Mamba%20for%20Object%20Detection.pdf)

## Real-time detectors

| Key | Title | File | Source or note |
|---|---|---|---|
| `lin2017fpn` | Feature Pyramid Networks for Object Detection | [pdf](01_detectors/2016_FPN_1612.03144.pdf) | https://arxiv.org/abs/1612.03144 |
| `liu2018panet` | Path Aggregation Network for Instance Segmentation | [pdf](01_detectors/2018_PANet_1803.01534.pdf) | https://arxiv.org/abs/1803.01534 |
| `zong2023codetr` | DETRs with Collaborative Hybrid Assignments Training | [pdf](01_detectors/2022_Co-DETR_2211.12860.pdf) | https://arxiv.org/abs/2211.12860 |
| `lyu2022rtmdet` | RTMDet: An Empirical Study of Designing Real-Time Object Detectors | [pdf](01_detectors/2022_RTMDet_2212.07784.pdf) | https://arxiv.org/abs/2212.07784 |
| `zhao2024rtdetr` | DETRs Beat YOLOs on Real-time Object Detection | [pdf](01_detectors/2023_RT-DETR_2304.08069.pdf) | https://arxiv.org/abs/2304.08069 |
| `peng2025dfine` | D-FINE: Redefine Regression Task in DETRs as Fine-grained Distribution Refinement | [pdf](01_detectors/2024_D-FINE_2410.13842.pdf) | https://arxiv.org/abs/2410.13842 |
| `huang2025deim` | DEIM: DETR with Improved Matching for Fast Convergence | [pdf](01_detectors/2024_DEIM_2412.04234.pdf) | https://arxiv.org/abs/2412.04234 |
| `wang2024yolov10` | YOLOv10: Real-Time End-to-End Object Detection | [pdf](01_detectors/2024_YOLOv10_2405.14458.pdf) | https://arxiv.org/abs/2405.14458 |
| `deimv2_2025` | Real-Time Object Detection Meets DINOv3 | [pdf](01_detectors/2025_DEIMv2_2509.20787.pdf) | https://arxiv.org/abs/2509.20787 |
| `robinson2025rfdetr` | RF-DETR: Neural Architecture Search for Real-Time Detection Transformers | [pdf](01_detectors/2025_RF-DETR_2511.09554.pdf) | https://arxiv.org/abs/2511.09554 |
| `rtdetrv4_2025` | RT-DETRv4: Painlessly Furthering Real-Time Object Detection with Vision Foundation Models | [pdf](01_detectors/2025_RT-DETRv4_2510.25257.pdf) | https://arxiv.org/abs/2510.25257 |
| `tian2025yolov12` | YOLOv12: Attention-Centric Real-Time Object Detectors | [pdf](01_detectors/2025_YOLOv12_2502.12524.pdf) | https://arxiv.org/abs/2502.12524 |
| `lei2025yolov13` | YOLOv13: Real-Time Object Detection with Hypergraph-Enhanced Adaptive Visual Perception | [pdf](01_detectors/2025_YOLOv13_2506.17733.pdf) | https://arxiv.org/abs/2506.17733 |
| `yolo26_2026` | Ultralytics YOLO26: Unified Real-Time End-to-End Vision Models | [pdf](01_detectors/2026_YOLO26_2606.03748.pdf) | https://arxiv.org/abs/2606.03748 |
| `yolo11_2024` | YOLO11 | not stored | software release, no paper: https://github.com/ultralytics/ultralytics |

## State-space models and vision SSMs

| Key | Title | File | Source or note |
|---|---|---|---|
| `gu2022s4` | Efficiently Modeling Long Sequences with Structured State Spaces | [pdf](02_state_space_models/2021_S4_2111.00396.pdf) | https://arxiv.org/abs/2111.00396 |
| `smith2023s5` | Simplified State Space Layers for Sequence Modeling | [pdf](02_state_space_models/2022_S5_2208.04933.pdf) | https://arxiv.org/abs/2208.04933 |
| `dao2023flash2` | FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning | [pdf](02_state_space_models/2023_FlashAttention-2_2307.08691.pdf) | https://arxiv.org/abs/2307.08691 |
| `gu2023mamba` | Mamba: Linear-Time Sequence Modeling with Selective State Spaces | [pdf](02_state_space_models/2023_Mamba_2312.00752.pdf) | https://arxiv.org/abs/2312.00752 |
| `huang2024localmamba` | LocalMamba: Visual State Space Model with Windowed Selective Scan | [pdf](02_state_space_models/2024_LocalMamba_2403.09338.pdf) | https://arxiv.org/abs/2403.09338 |
| `han2024mlla` | Demystify Mamba in Vision: A Linear Attention Perspective | [pdf](02_state_space_models/2024_MLLA_2405.16605.pdf) | https://arxiv.org/abs/2405.16605 |
| `dao2024mamba2` | Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality | [pdf](02_state_space_models/2024_Mamba-2_2405.21060.pdf) | https://arxiv.org/abs/2405.21060 |
| `yu2025mambaout` | MambaOut: Do We Really Need Mamba for Vision? | [pdf](02_state_space_models/2024_MambaOut_2405.07992.pdf) | https://arxiv.org/abs/2405.07992 |
| `hatamizadeh2025mambavision` | MambaVision: A Hybrid Mamba-Transformer Vision Backbone | [pdf](02_state_space_models/2024_MambaVision_2407.08083.pdf) | https://arxiv.org/abs/2407.08083 |
| `xie2024quadmamba` | QuadMamba: Learning Quadtree-based Selective Scan for Visual State Space Model | [pdf](02_state_space_models/2024_QuadMamba_2410.06806.pdf) | https://arxiv.org/abs/2410.06806 |
| `xiao2025spatialmamba` | Spatial-Mamba: Effective Visual State Space Models via Structure-aware State Fusion | [pdf](02_state_space_models/2024_Spatial-Mamba_2410.15091.pdf) | https://arxiv.org/abs/2410.15091 |
| `liu2024vmamba` | VMamba: Visual State Space Model | [pdf](02_state_space_models/2024_VMamba_2401.10166.pdf) | https://arxiv.org/abs/2401.10166 |
| `zhu2024vim` | Vision Mamba: Efficient Visual Representation Learning with Bidirectional State Space Model | [pdf](02_state_space_models/2024_Vim_2401.09417.pdf) | https://arxiv.org/abs/2401.09417 |
| `a2mamba2025` | A2Mamba: Attention-augmented State Space Models for Visual Recognition | [pdf](02_state_space_models/2025_A2Mamba_2507.16624.pdf) | https://arxiv.org/abs/2507.16624 |
| `damamba2025` | DAMamba: Vision State Space Model with Dynamic Adaptive Scan | [pdf](02_state_space_models/2025_DAMamba_2502.12627.pdf) | https://arxiv.org/abs/2502.12627 |
| `cgspn2026` | Scaling Parallel Sequence Models to Foundation-Scale Vision Encoders | [pdf](02_state_space_models/2026_C-GSPN_2606.00746.pdf) | https://arxiv.org/abs/2606.00746 |
| `graphscan2026` | Can Graphs Help Vision SSMs See Better? | [pdf](02_state_space_models/2026_GraphScan_2605.11300.pdf) | https://arxiv.org/abs/2605.11300 |
| `prismamba2026` | Partial Ring Scan: Revisiting Scan Order in Vision State Space Models | [pdf](02_state_space_models/2026_PRISMamba_2602.04170.pdf) | https://arxiv.org/abs/2602.04170 |
| `sfmamba2026` | SF-Mamba: Rethinking State Space Model for Vision | [pdf](02_state_space_models/2026_SF-Mamba_2603.16423.pdf) | https://arxiv.org/abs/2603.16423 |
| `vnct2026` | Vision Non-Causal Trapezoidal Mamba: Eliminating Directional Scanning in Vision SSMs with Second-Order Dynamics | [pdf](02_state_space_models/2026_VNCT_2607.03589.pdf) | https://arxiv.org/abs/2607.03589 |
| `mambapy_repo` | mamba.py | not stored | software: https://github.com/alxndrTL/mamba.py |
| `mambassm_repo` | mamba_ssm | not stored | software: https://github.com/state-spaces/mamba |
| `objm2025` | ObjM | not stored | no arXiv match |

## Mamba-based detectors

| Key | Title | File | Source or note |
|---|---|---|---|
| `wang2025mambayolo` | Mamba YOLO: A Simple Baseline for Object Detection with State Space Model | [pdf](03_mamba_detectors/2024_Mamba-YOLO_2406.05835.pdf) | https://arxiv.org/abs/2406.05835 |
| `mambanextyolo2025` | MambaNeXt-YOLO: A Hybrid State Space Model for Real-time Object Detection | [pdf](03_mamba_detectors/2025_MambaNeXt-YOLO_2506.03654.pdf) | https://arxiv.org/abs/2506.03654 |
| `hdmambayolo2026` | HDMamba-YOLO: Efficient State-Space Perception and Local Spatial Reconstruction for UAV Small Object | [pdf](03_mamba_detectors/2026_HDMamba-YOLO_2609.23061.pdf) | https://arxiv.org/abs/2609.23061 |
| `lcmamnet2026` | LCMamNet: A Lightweight Cross-scale Mamba Network for Infrared Small Target Detection | [pdf](03_mamba_detectors/2026_LCMamNet_2607.24184.pdf) | https://arxiv.org/abs/2607.24184 |
| `mambapsa2026` | MambaPSA: A Mamba-based Replacement for C2PSA in YOLO26 | [pdf](03_mamba_detectors/2026_MambaPSA_2607.12681.pdf) | https://arxiv.org/abs/2607.12681 |
| `scopemamba2026` | ScopeMamba-YOLO: Widening the Perceptual Scope Inward and Outward for Small Object Detection in Remote Sensing Imagery | [pdf](03_mamba_detectors/2026_ScopeMamba-YOLO_2609.10156.pdf) | https://arxiv.org/abs/2609.10156 |
| `yolo12mambascan2026` | YOLO12-MambaScan: An Efficient Object Detector with High-Frequency Enhancement and State-Space Modeling | [pdf](03_mamba_detectors/2026_YOLO12-MambaScan_2609.13647.pdf) | https://arxiv.org/abs/2609.13647 |
| `akcmamba2026` | AKCMamba-YOLO | not stored | no arXiv match |

## Small and tiny object detection

| Key | Title | File | Source or note |
|---|---|---|---|
| `yu2020tinyperson` | Scale Match for Tiny Person Detection | [pdf](04_small_objects/2019_TinyPerson-ScaleMatch_1912.10664.pdf) | https://arxiv.org/abs/1912.10664 |
| `wang2021nwd` | A Normalized Gaussian Wasserstein Distance for Tiny Object Detection | [pdf](04_small_objects/2021_NWD_2110.13389.pdf) | https://arxiv.org/abs/2110.13389 |
| `zhu2021tph` | TPH-YOLOv5: Improved YOLOv5 Based on Transformer Prediction Head for Object Detection on Drone-captured Scenarios | [pdf](04_small_objects/2021_TPH-YOLOv5_2108.11539.pdf) | https://arxiv.org/abs/2108.11539 |
| `xu2022aitodv2` | Detecting tiny objects in aerial images: A normalized Wasserstein distance and a new benchmark | [pdf](04_small_objects/2022_AI-TODv2-NWD_2206.13996.pdf) | https://arxiv.org/abs/2206.13996 |
| `akyon2022sahi` | Slicing Aided Hyper Inference and Fine-tuning for Small Object Detection | [pdf](04_small_objects/2022_SAHI_2202.06934.pdf) | https://arxiv.org/abs/2202.06934 |
| `cheng2023soda` | Towards Large-Scale Small Object Detection: Survey and Benchmarks | [pdf](04_small_objects/2022_SODA_2207.14096.pdf) | https://arxiv.org/abs/2207.14096 |
| `huang2024dqdetr` | DQ-DETR: DETR with Dynamic Query for Tiny Object Detection | [pdf](04_small_objects/2024_DQ-DETR_2404.03507.pdf) | https://arxiv.org/abs/2404.03507 |
| `domedetr2025` | Dome-DETR: DETR with Density-Oriented Feature-Query Manipulation for Efficient Tiny Object Detection | [pdf](04_small_objects/2025_Dome-DETR_2505.05741.pdf) | https://arxiv.org/abs/2505.05741 |
| `yuan2026stripr` | Strip R-CNN: Large Strip Convolution for Remote Sensing Object Detection | [pdf](04_small_objects/2025_Strip-RCNN_2501.03775.pdf) | https://arxiv.org/abs/2501.03775 |
| `uavdetr2025` | UAV-DETR: Efficient End-to-End Object Detection for Unmanned Aerial Vehicle Imagery | [pdf](04_small_objects/2025_UAV-DETR_2501.01855.pdf) | https://arxiv.org/abs/2501.01855 |
| `dernet2026` | From Spatial to Spectral: An Efficient, Frequency-Guided Feature Representation Learner for Small Object Detection | [pdf](04_small_objects/2026_DERNet_2606.23825.pdf) | https://arxiv.org/abs/2606.23825 |
| `efsidetr2026` | EFSI-DETR: Efficient Frequency-Semantic Integration for Real-Time Small Object Detection in UAV Imagery | [pdf](04_small_objects/2026_EFSI-DETR_2601.18597.pdf) | https://arxiv.org/abs/2601.18597 |
| `o2deim2026` | Real-Time Oriented Object Detection Transformer in Remote Sensing Images | [pdf](04_small_objects/2026_O2-DEIM_2603.15497.pdf) | https://arxiv.org/abs/2603.15497 |
| `sfdnet2026` | Adaptive Spectrum-Aware Feature Disentangled Network for Small Object Detection | [pdf](04_small_objects/2026_SFDNet_2606.29029.pdf) | https://arxiv.org/abs/2606.29029 |
| `tolf2026` | Noise-Robust Tiny Object Localization with Flows | [pdf](04_small_objects/2026_TOLF_2601.00617.pdf) | https://arxiv.org/abs/2601.00617 |
| `wang2020aitod` | AI-TOD | not stored | ICPR 2020; no open-access version found (title search returns a different 2024 paper) |
| `wang2025hpsdetr` | HPS-DETR | not stored | IEEE TGRS, not open access; the copy supplied is in 00_source_papers |
| `yang2026uavdet` | UAVDet | not stored | CVIU, not open access; the copy supplied is in 00_source_papers |

## Visible-thermal (RGB-T) fusion

| Key | Title | File | Source or note |
|---|---|---|---|
| `qingyun2021cft` | Cross-Modality Fusion Transformer for Multispectral Object Detection | [pdf](05_rgbt_fusion/2021_CFT_2111.00273.pdf) | https://arxiv.org/abs/2111.00273 |
| `chen2022proben` | Multimodal Object Detection via Probabilistic Ensembling | [pdf](05_rgbt_fusion/2021_ProbEn_2104.02904.pdf) | https://arxiv.org/abs/2104.02904 |
| `shen2024icafusion` | ICAFusion: Iterative Cross-Attention Guided Feature Fusion for Multispectral Object Detection | [pdf](05_rgbt_fusion/2023_ICAFusion_2308.07504.pdf) | https://arxiv.org/abs/2308.07504 |
| `li2025cfmw` | CFMW: Cross-modality Fusion Mamba for Robust Object Detection under Adverse Weather | [pdf](05_rgbt_fusion/2024_CFMW-preprint_2404.16302.pdf) | https://arxiv.org/abs/2404.16302 |
| `como2024` | COMO: Cross-Mamba Interaction and Offset-Guided Fusion for Multimodal Object Detection | [pdf](05_rgbt_fusion/2024_COMO_2412.18076.pdf) | https://arxiv.org/abs/2412.18076 |
| `dong2025fusionmamba` | Fusion-Mamba for Cross-modality Object Detection | [pdf](05_rgbt_fusion/2024_Fusion-Mamba_2404.09146.pdf) | https://arxiv.org/abs/2404.09146 |
| `mambast2024` | MambaST: A Plug-and-Play Cross-Spectral Spatial-Temporal Fuser for Efficient Pedestrian Detection | [pdf](05_rgbt_fusion/2024_MambaST_2408.01037.pdf) | https://arxiv.org/abs/2408.01037 |
| `remotedetmamba2024` | RemoteDet-Mamba: A Hybrid Mamba-CNN Network for Multi-modal Object Detection in Remote Sensing Images | [pdf](05_rgbt_fusion/2024_RemoteDet-Mamba_2410.13532.pdf) | https://arxiv.org/abs/2410.13532 |
| `cbcswca2025` | Mitigating the Impact of Prominent Position Shift in Drone-based RGBT Object Detection | [pdf](05_rgbt_fusion/2025_CBC-SWCA-position-shift_2502.09311.pdf) | https://arxiv.org/abs/2502.09311 |
| `ms2fusion2025` | Multispectral State-Space Feature Fusion: Bridging Shared and Cross-Parametric Interactions for Object Detection | [pdf](05_rgbt_fusion/2025_MS2Fusion_2507.14643.pdf) | https://arxiv.org/abs/2507.14643 |
| `mambarefine2025` | MambaRefine-YOLO: A Dual-Modality Small Object Detector for UAV Imagery | [pdf](05_rgbt_fusion/2025_MambaRefine-YOLO_2511.19134.pdf) | https://arxiv.org/abs/2511.19134 |
| `wavemamba2025` | WaveMamba: Wavelet-Driven Mamba Fusion for RGB-Infrared Object Detection | [pdf](05_rgbt_fusion/2025_WaveMamba_2507.18173.pdf) | https://arxiv.org/abs/2507.18173 |
| `cfgpnet2026` | CFGPNet: Cross-Attention-Based Fused Gradient Programmed Network Framework for Multispectral Object Detection | [pdf](05_rgbt_fusion/2026_CFGPNet_2608.06205.pdf) | https://arxiv.org/abs/2608.06205 |
| `dlrmamba2026` | DLRMamba: Distilling Low-Rank Mamba for Edge Multispectral Fusion Object Detection | [pdf](05_rgbt_fusion/2026_DLRMamba_2603.06920.pdf) | https://arxiv.org/abs/2603.06920 |
| `registerbridge2026` | RegisterBridgeMM: A Register-Centric Framework for RGB-Infrared Object Detection | [pdf](05_rgbt_fusion/2026_RegisterBridgeMM_2608.04833.pdf) | https://arxiv.org/abs/2608.04833 |
| `cgssm2025` | CG-SSM-misaligned-fusion | not stored | no arXiv match |
| `msdfmamba2025` | MSDF-Mamba | not stored | no arXiv match |

## Video detection, tracking and streaming

| Key | Title | File | Source or note |
|---|---|---|---|
| `ballas2016convgru` | Delving Deeper into Convolutional Networks for Learning Video Representations | [pdf](06_video_tracking_streaming/2015_ConvGRU_1511.06432.pdf) | https://arxiv.org/abs/1511.06432 |
| `ristani2016idf1` | Performance Measures and a Data Set for Multi-Target, Multi-Camera Tracking | [pdf](06_video_tracking_streaming/2016_IDF1-performance-measures_1609.01775.pdf) | https://arxiv.org/abs/1609.01775 |
| `luiten2021hota` | HOTA: A Higher Order Metric for Evaluating Multi-Object Tracking | [pdf](06_video_tracking_streaming/2020_HOTA_2009.07736.pdf) | https://arxiv.org/abs/2009.07736 |
| `li2020streaming` | Towards Streaming Perception | [pdf](06_video_tracking_streaming/2020_Towards-Streaming-Perception_2005.10420.pdf) | https://arxiv.org/abs/2005.10420 |
| `zhang2022bytetrack` | ByteTrack: Multi-Object Tracking by Associating Every Detection Box | [pdf](06_video_tracking_streaming/2021_ByteTrack_2110.06864.pdf) | https://arxiv.org/abs/2110.06864 |
| `aharon2022botsort` | BoT-SORT: Robust Associations Multi-Pedestrian Tracking | [pdf](06_video_tracking_streaming/2022_BoT-SORT_2206.14651.pdf) | https://arxiv.org/abs/2206.14651 |
| `li2023longshortnet` | LongShortNet: Exploring Temporal and Semantic Features Fusion in Streaming Perception | [pdf](06_video_tracking_streaming/2022_LongShortNet_2210.15518.pdf) | https://arxiv.org/abs/2210.15518 |
| `zhang2023motrv2` | MOTRv2: Bootstrapping End-to-End Multi-Object Tracking by Pretrained Object Detectors | [pdf](06_video_tracking_streaming/2022_MOTRv2_2211.09791.pdf) | https://arxiv.org/abs/2211.09791 |
| `cao2023ocsort` | Observation-Centric SORT: Rethinking SORT for Robust Multi-Object Tracking | [pdf](06_video_tracking_streaming/2022_OC-SORT_2203.14360.pdf) | https://arxiv.org/abs/2203.14360 |
| `gehrig2023rvt` | Recurrent Vision Transformers for Object Detection with Event Cameras | [pdf](06_video_tracking_streaming/2022_RVT_2212.05598.pdf) | https://arxiv.org/abs/2212.05598 |
| `yang2022streamyolo` | Real-time Object Detection for Streaming Perception | [pdf](06_video_tracking_streaming/2022_StreamYOLO_2203.12338.pdf) | https://arxiv.org/abs/2203.12338 |
| `he2023damostreamnet` | DAMO-StreamNet: Optimizing Streaming Perception in Autonomous Driving | [pdf](06_video_tracking_streaming/2023_DAMO-StreamNet_2303.17144.pdf) | https://arxiv.org/abs/2303.17144 |
| `einet2023` | Erasure-based Interaction Network for RGBT Video Object Detection and A Unified Benchmark | [pdf](06_video_tracking_streaming/2023_EINet-VT-VOD50_2308.01630.pdf) | https://arxiv.org/abs/2308.01630 |
| `hgttrack2024` | Heterogeneous Graph Transformer for Multiple Tiny Object Tracking in RGB-T Videos | [pdf](06_video_tracking_streaming/2024_HGT-Track-VT-Tiny-MOT_2412.10861.pdf) | https://arxiv.org/abs/2412.10861 |
| `gao2025motip` | Multiple Object Tracking as ID Prediction | [pdf](06_video_tracking_streaming/2024_MOTIP_2403.16848.pdf) | https://arxiv.org/abs/2403.16848 |
| `msenet2024` | Mixture of Scale Experts for Alignment-free RGBT Video Object Detection and A Unified Benchmark | [pdf](06_video_tracking_streaming/2024_MSENet-UVT-VOD2024_2410.12143.pdf) | https://arxiv.org/abs/2410.12143 |
| `mambatrack2024` | MambaTrack: A Simple Baseline for Multiple Object Tracking with State Space Model | [pdf](06_video_tracking_streaming/2024_MambaTrack_2408.09178.pdf) | https://arxiv.org/abs/2408.09178 |
| `zubic2024ssmevent` | State Space Models for Event Cameras | [pdf](06_video_tracking_streaming/2024_SSMs-for-event-cameras_2402.15584.pdf) | https://arxiv.org/abs/2402.15584 |
| `segu2025sambamotr` | Samba: Synchronized Set-of-Sequences Modeling for Multiple Object Tracking | [pdf](06_video_tracking_streaming/2024_SambaMOTR_2410.01806.pdf) | https://arxiv.org/abs/2410.01806 |
| `trackssm2024` | TrackSSM: A General Motion Predictor by State-Space Model | [pdf](06_video_tracking_streaming/2024_TrackSSM_2409.00487.pdf) | https://arxiv.org/abs/2409.00487 |
| `matr2025` | Motion-Aware Transformer for Multi-Object Tracking | [pdf](06_video_tracking_streaming/2025_MATR_2509.21715.pdf) | https://arxiv.org/abs/2509.21715 |
| `msgnet2025` | Multimodal Spatio-temporal Graph Learning for Alignment-free RGBT Video Object Detection | [pdf](06_video_tracking_streaming/2025_MSGNet_2504.11779.pdf) | https://arxiv.org/abs/2504.11779 |
| `shim2025tracktrack` | Focusing on Tracks for Online Multi-Object Tracking | [pdf](06_video_tracking_streaming/2025_TrackTrack.pdf) | https://openaccess.thecvf.com/content/CVPR2025/papers/Shim_Focusing_on_Tracks_for_Online_Multi-Object_Tracking_CVPR_2025_paper.pdf |
| `dchnet2026` | Dual-Correlation Hypergraph Network for Unaligned RGBT Video Object Detection and A Large-scale Benchmark | [pdf](06_video_tracking_streaming/2026_DCHNet-DVT-VOD1000_2607.08191.pdf) | https://arxiv.org/abs/2607.08191 |
| `smp2026` | Selective Mask Propagation for Multi-Object Tracking | [pdf](06_video_tracking_streaming/2026_SAM3-mask-propagation_2606.13033.pdf) | https://arxiv.org/abs/2606.13033 |
| `scdt2026` | Spatio-Temporal Conditional Denoising Transformer for Modality-Missing RGBT Tracking | [pdf](06_video_tracking_streaming/2026_SCDT_2607.24701.pdf) | https://arxiv.org/abs/2607.24701 |
| `spikingmot2026` | SpikingMOT: A Spike-Driven Multi-Object Tracker | [pdf](06_video_tracking_streaming/2026_SpikingMOT_2607.19875.pdf) | https://arxiv.org/abs/2607.19875 |
| `deepstream_docs` | DeepStream | not stored | documentation: https://docs.nvidia.com/metropolis/deepstream |
| `fastrtc_repo` | FastRTC | not stored | software: https://github.com/gradio-app/fastrtc |
| `gradio_repo` | Gradio | not stored | software: https://github.com/gradio-app/gradio |
| `pyav_repo` | PyAV | not stored | software: https://github.com/PyAV-Org/PyAV |
| `roboflowtrackers_repo` | trackers | not stored | software: https://github.com/roboflow/trackers |
| `tmambadet2026` | TMambaDet | not stored | no arXiv match |
| `trackeval_repo` | TrackEval | not stored | software: https://github.com/JonathonLuiten/TrackEval |
| `williams1990tbptt` | TBPTT | not stored | Neural Computation 2(4), 1990; not open access |

## Datasets and benchmarks

| Key | Title | File | Source or note |
|---|---|---|---|
| `everingham2010voc` | The PASCAL Visual Object Classes (VOC) Challenge | [pdf](07_datasets/2010_PASCAL-VOC.pdf) | http://host.robots.ox.ac.uk/pascal/VOC/pubs/everingham10.pdf |
| `lin2014coco` | Microsoft COCO: Common Objects in Context | [pdf](07_datasets/2014_COCO_1405.0312.pdf) | https://arxiv.org/abs/1405.0312 |
| `russakovsky2015imagenet` | ImageNet Large Scale Visual Recognition Challenge | [pdf](07_datasets/2014_ImageNet-ILSVRC_1409.0575.pdf) | https://arxiv.org/abs/1409.0575 |
| `hwang2015kaist` | Multispectral Pedestrian Detection: Benchmark Dataset and Baseline | [pdf](07_datasets/2015_KAIST.pdf) | https://openaccess.thecvf.com/content_cvpr_2015/papers/Hwang_Multispectral_Pedestrian_Detection_2015_CVPR_paper.pdf |
| `milan2016mot16` | MOT16: A Benchmark for Multi-Object Tracking | [pdf](07_datasets/2016_MOT16-MOT17_1603.00831.pdf) | https://arxiv.org/abs/1603.00831 |
| `xia2018dota` | DOTA: A Large-scale Dataset for Object Detection in Aerial Images | [pdf](07_datasets/2017_DOTA_1711.10398.pdf) | https://arxiv.org/abs/1711.10398 |
| `yu2020bdd100k` | BDD100K: A Diverse Driving Dataset for Heterogeneous Multitask Learning | [pdf](07_datasets/2018_BDD100K_1805.04687.pdf) | https://arxiv.org/abs/1805.04687 |
| `li2018kaistsanitized` | Multispectral Pedestrian Detection via Simultaneous Detection and Segmentation | [pdf](07_datasets/2018_KAIST-sanitized_1808.04818.pdf) | https://arxiv.org/abs/1808.04818 |
| `fan2019lasot` | LaSOT: A High-quality Benchmark for Large-scale Single Object Tracking | [pdf](07_datasets/2018_LaSOT_1809.07845.pdf) | https://arxiv.org/abs/1809.07845 |
| `kuznetsova2020openimages` | The Open Images Dataset V4: Unified image classification, object detection, and visual relationship detection at scale | [pdf](07_datasets/2018_OpenImages-V4_1811.00982.pdf) | https://arxiv.org/abs/1811.00982 |
| `du2018uavdt` | The Unmanned Aerial Vehicle Benchmark: Object Detection and Tracking | [pdf](07_datasets/2018_UAVDT_1804.00518.pdf) | https://arxiv.org/abs/1804.00518 |
| `lam2018xview` | xView: Objects in Context in Overhead Imagery | [pdf](07_datasets/2018_xView_1802.07856.pdf) | https://arxiv.org/abs/1802.07856 |
| `gupta2019lvis` | LVIS: A Dataset for Large Vocabulary Instance Segmentation | [pdf](07_datasets/2019_LVIS_1908.03195.pdf) | https://arxiv.org/abs/1908.03195 |
| `shao2019objects365` | Objects365: A Large-Scale, High-Quality Dataset for Object Detection | [pdf](07_datasets/2019_Objects365.pdf) | https://openaccess.thecvf.com/content_ICCV_2019/papers/Shao_Objects365_A_Large-Scale_High-Quality_Dataset_for_Object_Detection_ICCV_2019_paper.pdf |
| `sun2022dronevehicle` | Drone-based RGB-Infrared Cross-Modality Vehicle Detection via Uncertainty-Aware Learning | [pdf](07_datasets/2020_DroneVehicle_2003.02437.pdf) | https://arxiv.org/abs/2003.02437 |
| `zhang2020cfr` | Multispectral Fusion for Object Detection with Cyclic Fuse-and-Refine Blocks | [pdf](07_datasets/2020_FLIR-aligned-CFR_2009.12664.pdf) | https://arxiv.org/abs/2009.12664 |
| `corona2021meva` | MEVA: A Large-Scale Multiview, Multimodal Video Dataset for Activity Detection | [pdf](07_datasets/2020_MEVA_2012.00914.pdf) | https://arxiv.org/abs/2012.00914 |
| `dendorfer2020mot20` | MOT20: A benchmark for multi object tracking in crowded scenes | [pdf](07_datasets/2020_MOT20_2003.09003.pdf) | https://arxiv.org/abs/2003.09003 |
| `dave2020tao` | TAO: A Large-Scale Benchmark for Tracking Any Object | [pdf](07_datasets/2020_TAO_2005.10356.pdf) | https://arxiv.org/abs/2005.10356 |
| `zhu2021visdrone` | Detection and Tracking Meet Drones Challenge | [pdf](07_datasets/2020_VisDrone_2001.06303.pdf) | https://arxiv.org/abs/2001.06303 |
| `ding2021dotav2` | Object Detection in Aerial Images: A Large-Scale Benchmark and Challenges | [pdf](07_datasets/2021_DOTA-v2_2102.12219.pdf) | https://arxiv.org/abs/2102.12219 |
| `sun2022dancetrack` | DanceTrack: Multi-Object Tracking in Uniform Appearance and Diverse Motion | [pdf](07_datasets/2021_DanceTrack_2111.14690.pdf) | https://arxiv.org/abs/2111.14690 |
| `jia2021llvip` | LLVIP: A Visible-infrared Paired Dataset for Low-light Vision | [pdf](07_datasets/2021_LLVIP_2108.10831.pdf) | https://arxiv.org/abs/2108.10831 |
| `varga2022seadronessee` | SeaDronesSee: A Maritime Benchmark for Detecting Humans in Open Water | [pdf](07_datasets/2021_SeaDronesSee_2105.01922.pdf) | https://arxiv.org/abs/2105.01922 |
| `liu2022tardal` | Target-aware Dual Adversarial Learning and a Multi-scenario Multi-Modality Benchmark to Fuse Infrared and Visible for Object Detection | [pdf](07_datasets/2022_M3FD-TarDAL_2203.16220.pdf) | https://arxiv.org/abs/2203.16220 |
| `cui2023sportsmot` | SportsMOT: A Large Multi-Object Tracking Dataset in Multiple Sports Scenes | [pdf](07_datasets/2023_SportsMOT_2304.05170.pdf) | https://arxiv.org/abs/2304.05170 |
| `singh2024cocorem` | Benchmarking Object Detectors with COCO: A New Path Forward | [pdf](07_datasets/2024_COCO-ReM_2403.18819.pdf) | https://arxiv.org/abs/2403.18819 |
| `ying2025rgbttiny` | Visible-Thermal Tiny Object Detection: A Benchmark Dataset and Baselines | [pdf](07_datasets/2024_RGBT-Tiny_2406.14482.pdf) | https://arxiv.org/abs/2406.14482 |
| `vtmot2024` | Visible-Thermal Multiple Object Tracking: Large-scale Video Dataset and Progressive Fusion Approach | [pdf](07_datasets/2024_VT-MOT_2408.00969.pdf) | https://arxiv.org/abs/2408.00969 |
| `xsvid2024` | XS-VID: An Extremely Small Video Object Detection Dataset | [pdf](07_datasets/2024_XS-VID_2407.18137.pdf) | https://arxiv.org/abs/2407.18137 |
| `mmot2025` | MMOT: The First Challenging Benchmark for Drone-based Multispectral Multi-Object Tracking | [pdf](07_datasets/2025_MMOT_2510.12565.pdf) | https://arxiv.org/abs/2510.12565 |
| `robicheaux2025rf100vl` | Roboflow100-VL: A Multi-Domain Object Detection Benchmark for Vision-Language Models | [pdf](07_datasets/2025_RF100-VL_2505.20612.pdf) | https://arxiv.org/abs/2505.20612 |
| `drgbt2026` | DRGBT-1K: A Large-scale High-quality Benchmark for Dynamic RGBT Tracking | [pdf](07_datasets/2026_DRGBT-1K_2607.19772.pdf) | https://arxiv.org/abs/2607.19772 |
| `tinyset2026` | Generalized Small Object Detection:A Point-Prompted Paradigm and Benchmark | [pdf](07_datasets/2026_TinySet-9M_2604.02773.pdf) | https://arxiv.org/abs/2604.02773 |
| `fastercocoeval_repo` | faster-coco-eval | not stored | software: https://github.com/MiXaiLL76/faster_coco_eval |
| `gonzalez2016cvc14` | CVC-14 | not stored | open access but the publisher blocks scripted downloads; download from https://www.mdpi.com/1424-8220/16/6/820 |
| `m3ot2025` | M3OT | not stored | open access but the publisher blocks scripted downloads; download from https://www.nature.com/articles/s41597-025-06204-0 |
| `razakarivony2016vedai` | VEDAI | not stored | J. Visual Communication and Image Representation 2016; not on arXiv; dataset page: https://downloads.greyc.fr/vedai/ |

## Foundation models

| Key | Title | File | Source or note |
|---|---|---|---|
| `oquab2024dinov2` | DINOv2: Learning Robust Visual Features without Supervision | [pdf](08_foundation_models/2023_DINOv2_2304.07193.pdf) | https://arxiv.org/abs/2304.07193 |
| `simeoni2025dinov3` | DINOv3 | [pdf](08_foundation_models/2025_DINOv3_2508.10104.pdf) | https://arxiv.org/abs/2508.10104 |
| `sam3_2025` | SAM 3: Segment Anything with Concepts | [pdf](08_foundation_models/2025_SAM3_2511.16719.pdf) | https://arxiv.org/abs/2511.16719 |
| `wang2025yoloe` | YOLOE: Real-Time Seeing Anything | [pdf](08_foundation_models/2025_YOLOE_2503.07465.pdf) | https://arxiv.org/abs/2503.07465 |

## Other methods (blocks, losses, diffusion)

| Key | Title | File | Source or note |
|---|---|---|---|
| `song2021ddim` | Denoising Diffusion Implicit Models | [pdf](09_other_methods/2020_DDIM_2010.02502.pdf) | https://arxiv.org/abs/2010.02502 |
| `ho2020ddpm` | Denoising Diffusion Probabilistic Models | [pdf](09_other_methods/2020_DDPM_2006.11239.pdf) | https://arxiv.org/abs/2006.11239 |
| `ozdenizci2023weatherdiff` | Restoring Vision in Adverse Weather Conditions with Patch-Based Denoising Diffusion Models | [pdf](09_other_methods/2022_WeatherDiff_2207.14626.pdf) | https://arxiv.org/abs/2207.14626 |
| `kang2024asfyolo` | ASF-YOLO: A Novel YOLO Model with Attentional Scale Sequence Fusion for Cell Instance Segmentation | [pdf](09_other_methods/2023_ASF-YOLO_2312.06458.pdf) | https://arxiv.org/abs/2312.06458 |
| `liu2023dysample` | Learning to Upsample by Learning to Sample | [pdf](09_other_methods/2023_DySample_2308.15085.pdf) | https://arxiv.org/abs/2308.15085 |
| `liu2023efficientvit` | EfficientViT: Memory Efficient Vision Transformer with Cascaded Group Attention | [pdf](09_other_methods/2023_EfficientViT_2305.07027.pdf) | https://arxiv.org/abs/2305.07027 |
| `chen2023fasternet` | Run, Don't Walk: Chasing Higher FLOPS for Faster Neural Networks | [pdf](09_other_methods/2023_FasterNet_2303.03667.pdf) | https://arxiv.org/abs/2303.03667 |
| `zhang2023inneriou` | Inner-IoU: More Effective Intersection over Union Loss with Auxiliary Bounding Box | [pdf](09_other_methods/2023_Inner-IoU_2311.02877.pdf) | https://arxiv.org/abs/2311.02877 |
| `ma2023mpdiou` | MPDIoU: A Loss for Efficient and Accurate Bounding Box Regression | [pdf](09_other_methods/2023_MPDIoU_2307.07662.pdf) | https://arxiv.org/abs/2307.07662 |

## Law and ethics

| Key | Title | File | Source or note |
|---|---|---|---|
| `gdpr2016` | Regulation (EU) 2016/679 (General Data Protection Regulation) | [pdf](10_law_and_ethics/2016_GDPR-Regulation-2016-679.pdf) | https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=CELEX:32016R0679 |
| `peng2021stewardship` | Mitigating Dataset Harms Requires Stewardship: Lessons from 1000 Papers | [pdf](10_law_and_ethics/2021_Dataset-stewardship-1000-papers_2108.02922.pdf) | https://arxiv.org/abs/2108.02922 |
| `euaiact2024` | Regulation (EU) 2024/1689 (Artificial Intelligence Act) | [pdf](10_law_and_ethics/2024_EU-AI-Act-Regulation-2024-1689.pdf) | https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=CELEX:32024R1689 |
| `dpdp2023` | India-DPDP-Act-2023 | not stored | official text: https://www.meity.gov.in (Digital Personal Data Protection Act, 2023) |
| `dpdprules2025` | India-DPDP-Rules-2025 | not stored | notified November 2025; official text on meity.gov.in and egazette.gov.in |
| `poi2011` | Person-of-Interest | not stored | television series, not a paper |
| `puttaswamy2017` | Puttaswamy-2017 | not stored | Supreme Court of India judgment, (2017) 10 SCC 1 |
