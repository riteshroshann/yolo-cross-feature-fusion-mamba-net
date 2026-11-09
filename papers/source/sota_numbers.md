# Published RGB-T detector results on LLVIP, M3FD and KAIST

Compiled 2026-11-09 for the IEEE TIP paper. Every number below was read from the paper's own PDF (arXiv, CVF open access) unless the row says "secondary", in which case it is a number another paper printed for that method and I did not find it in the method's own paper. Citation keys refer to `refs_sota.bib`. mAP50-95 is written mAP; "-" means the paper did not report it. Table numbers are those of the PDF version I read (arXiv version named in the bib entry).

Not included because I could not read a number from an accessible primary source: ICAFusion on LLVIP and M3FD (the arXiv v1 reports only KAIST, FLIR, VEDAI), CIAN's own paper (Information Fusion, paywalled; its numbers come via GAFF and MBNet tables), and any number from IEEE Xplore or ScienceDirect pages I could not open.

Not included because they report none of the three datasets: SuperYOLO (VEDAI, DroneVehicle), YOLOFusion (VEDAI; DroneVehicle only in third-party tables), DMM (DroneVehicle, VEDAI). They are in the bib file for related work.

## LLVIP (12,025 train / 3,463 test pairs unless noted)

| Method | Venue | Backbone / detector | Input | mAP50 | mAP75 | mAP | Source |
|---|---|---|---|---|---|---|---|
| YOLOv5 visible | LLVIP paper (ICCVW 2021) | YOLOv5-l | not stated | 90.8 | 56.4 | 52.7 | jia2021llvip, Table 3 |
| YOLOv5 infrared | same | YOLOv5-l | not stated | 96.5 | 76.4 | 67.0 | jia2021llvip, Table 3 |
| YOLOv3 visible / infrared | same | YOLOv3 | not stated | 87.1 / 94.0 | 45.5 / 66.1 | 46.6 / 58.2 | jia2021llvip, Table 3 |
| CFT | arXiv 2021 (v4 2022) | YOLOv5, CFB/CSPD53 | 640x640 | 97.5 | 72.9 | 63.6 | fang2021cft, Table 4 (also Table 1) |
| CSSA (cited elsewhere as CSAA) | CVPRW 2023 | ResNet50, Faster R-CNN | 640x512 | 94.3 | 66.6 | 59.2 | cao2023csaa, Table 1 |
| GAFF re-run | WACV 2021 (numbers from CSSA) | not stated in CSSA table | 640x512 | 94.0 | 60.2 | 55.8 | cao2023csaa, Table 1 (secondary) |
| RSDet | arXiv 2024 (TITS 2026) | ResNet50, Faster R-CNN | resized to LLVIP resolution | 95.8 | - | 61.3 | zhao2026rsdet, Table VI |
| MS-DETR | IEEE TITS 2024 | ResNet50, deformable DETR | 640x512 (FLOPs setting) | 97.9 | 76.3 | 66.1 | xing2024msdetr, Table IV |
| Fusion-Mamba (YOLOv5) | arXiv 2024 (IEEE TMM 2025) | YOLOv5-l | 640x640 | 96.8 | - | 62.8 | dong2025fusionmamba, Table 1 |
| Fusion-Mamba (YOLOv8) | same | YOLOv8-l | 640x640 | 97.0 | - | 64.3 | dong2025fusionmamba, Table 1 |
| DAMSDet | ECCV 2024 | ResNet50, DETR | 1024x1024 | 97.9 | 79.1 | 69.6 | guo2024damsdet, Table 3 |
| GM-DETR (fus) | CVPRW 2024 | RT-DETR based | 640x640 | 97.5 | 80.6 | 69.6 | xiao2024gmdetr, Table 2 |
| GM-DETR (iso+fus) | same | RT-DETR based | 640x640 | 97.4 | 81.4 | 70.2 | xiao2024gmdetr, Table 2 |
| TFDet (Faster R-CNN) | IEEE TNNLS 2024 | ResNet50 | not stated | 96.0 | - | 59.4 | zhang2024tfdet, Table IV |
| TFDet (YOLOv5) | same | YOLOv5-l, 100 epochs | not stated | 97.9 | - | 71.1 | zhang2024tfdet, Table IV |
| YOLOv11x-RGBT-MCF | arXiv 2025 | YOLOv11x, COCO-pretrained | 640x640 | 97.06 | - | 70.26 | wan2025yolov11rgbt, Table 9 |
| WaveMamba (ResNet50) | ICCV 2025 | ResNet50 | not stated | 97.3 | - | 65.0 | zhu2025wavemamba, Table 3 |
| WaveMamba (YOLOv5) | same | YOLOv5 | not stated | 97.6 | - | 65.2 | zhu2025wavemamba, Table 3 |
| WaveMamba (YOLOv8) | same | YOLOv8 | not stated | 98.3 | - | 66.0 | zhu2025wavemamba, Table 3 |
| IRDFusion | arXiv 2025 | double Co-DETR | 640x640 | 98.4 | 83.1 | 70.9 | shen2025irdfusion, Table 2 |
| ICAFusion | Pattern Recognition 2024 | - | - | 98.4 | 76.2 | 64.5 | shen2025irdfusion, Table 2 (secondary; not in the ICAFusion arXiv paper, and WaveMamba Table 3 prints 95.2 / 60.1 for it) |

LLVIP caveats.
- The LLVIP split is the official 12,025 / 3,463 in every row above except the original LLVIP paper, whose arXiv text says 77.6 percent train and 22.4 percent test (the same ratio), and GM-DETR, RSDet and MS-DETR, which do not restate it in the text I read.
- mAP (50-95) depends on input size and detector family. DAMSDet trains at 1024x1024, the YOLO and Mamba papers at 640x640, CSSA at 640x512. Do not rank methods on mAP across input sizes without saying so.
- Infrared-only baselines are already strong: YOLOv5-l IR is 96.5 / 67.0 in the LLVIP paper, 94.6 / 61.9 in CFT Table 4, and 97.41 / 69.93 (YOLOv11x) in YOLOv11-RGBT Table 9. A fusion method that does not beat the same-family IR baseline at the same input size has not shown fusion gains. Several 2024 to 2025 fusion numbers (96.8 to 97.9 mAP50) sit within about one point of these baselines.
- CSSA, not CSAA, is the paper's own name for its module. Later papers cite it as CSAA, and WaveMamba Table 3 prints its mAP as 54.2 where the paper says 59.2.
- The LLVIP paper also reports log-average miss rate (Table 4): YOLOv5 visible 22.59, infrared 10.66; YOLOv3 visible 37.70, infrared 19.73 (percent).

## M3FD (no official split; protocols differ)

| Method | Venue | Backbone / detector | Input | Split train/test | mAP50 | mAP | Source |
|---|---|---|---|---|---|---|---|
| TarDAL (fused image + YOLOv5) | CVPR 2022 | YOLOv5 | not stated | 3,500 train images stated; test split not stated in the text I read | 80.7 (0.807) | not reported | liu2022tardal, Table 2 (all-scene column) |
| TarDAL infrared-only / visible-only | same | YOLOv5 | not stated | same | 78.1 / 75.6 | not reported | liu2022tardal, Table 2 |
| Fusion-Mamba (YOLOv5) | arXiv 2024 (TMM 2025) | YOLOv5-l | 640x640 | split of Liang et al. (RA-L 2023), counts not given | 85.0 | 57.5 | dong2025fusionmamba, Table 2 |
| Fusion-Mamba (YOLOv8) | same | YOLOv8-l | 640x640 | same | 88.0 | 61.9 | dong2025fusionmamba, Table 2 (row is garbled in text extraction; 88.0 / 61.9 matches the paper's stated gains over YOLOv5 and IRDFusion Table 3) |
| TarDAL re-run | same | YOLOv5 | 640x640 | same | 80.5 | 54.1 | dong2025fusionmamba, Table 2 (secondary) |
| DAMSDet | ECCV 2024 | ResNet50, DETR | 640x640 | 3,368 / 831 split by scene | 80.2 | 52.9 | guo2024damsdet, Table 1 |
| CFT re-run on DAMSDet split | arXiv 2021 | CSPDarknet53 | 640x640 | 3,368 / 831 | 68.2 | 42.5 | guo2024damsdet, Table 1 (secondary) |
| ICAFusion re-run on DAMSDet split | PR 2024 | CSPDarknet53 | 640x640 | 3,368 / 831 | 67.8 | 41.9 | guo2024damsdet, Table 1 (secondary) |
| TFDet | IEEE TNNLS 2024 | YOLOv5 | 640x640 letterbox | 2,905 / 1,295 (from arXiv:2405.16038) | 64.8 | 41.0 | zhang2024tfdet, Table VI |
| TarDAL re-run on TFDet split | CVPR 2022 | YOLOv5 | 640x640 | 2,905 / 1,295 | 61.9 | 39.1 | zhang2024tfdet, Table VI (secondary) |
| E2E-MFD | NeurIPS 2024 | Detectron2 end-to-end | not stated for E2E | 2,940 / 1,260 | 91.8 | 63.8 | zhang2024e2emfd, Table 2 |
| E2E-MFD fused images + YOLOv5s | same | YOLOv5s from scratch | 1024x1024 | 2,940 / 1,260 | 87.9 | 58.1 | zhang2024e2emfd, Table 2 |
| CFT and ICAFusion inside E2E-MFD's comparison | same | YOLOv5s | 1024x1024 | 2,940 / 1,260 | 89.8 / 87.4 | 60.7 / 57.2 | zhang2024e2emfd, Table 2 (secondary) |
| YOLOv11x RGB+IR (P3), from scratch | arXiv 2025 | YOLOv11x | 640x640 | 3,360 / 840 | 87.97 | 62.79 | wan2025yolov11rgbt, Table 11 |
| YOLOv11x-Midfusion, COCO-pretrained | same | YOLOv11x | 640x640 | 3,360 / 840 | 90.62 | 65.95 | wan2025yolov11rgbt, Table 12 |
| YOLOv11x-RGBT-MCF, COCO-pretrained | same | YOLOv11x | 640x640 | 3,360 / 840 | 89.83 | 64.23 | wan2025yolov11rgbt, Table 12 |
| WaveMamba (ResNet50 / YOLOv5 / YOLOv8) | ICCV 2025 | as named | not stated | 3,360 / 840 | 90.9 / 91.5 / 92.1 | 62.3 / 63.8 / 64.4 | zhu2025wavemamba, Table 1 |
| IRDFusion | arXiv 2025 | double Co-DETR, 36 epochs | 640x640 | 8:2 split, taken from its ref [41] (not resolved) | 90.8 | 61.9 | shen2025irdfusion, Table 3 |
| ICAFusion | PR 2024 | - | - | IRDFusion's split | 90.8 | 60.9 | shen2025irdfusion, Table 3 (secondary; ICAFusion arXiv has no M3FD) |

M3FD caveats. The numbers above are not comparable across rows.
- Split sizes seen in print: 3,500 train (TarDAL), 3,368 / 831 by scene (DAMSDet), 3,360 / 840 (YOLOv11-RGBT, WaveMamba), 2,940 / 1,260 (E2E-MFD), 2,905 / 1,295 (TFDet), 8:2 (IRDFusion), and an unstated count from Liang et al. (Fusion-Mamba). I did not resolve IRDFusion's split citation (its ref [41]), so its counts are unknown.
- The same method swings by more than 20 points of mAP50 with the split: CFT is 68.2 on DAMSDet's scene-disjoint split and 89.8 in E2E-MFD's table. DAMSDet states that its scene split is deliberately hard, with low similarity between train and validation scenes. TFDet's split also gives low absolute numbers (64.8 / 41.0 for the best method).
- Rows that use TarDAL-style fusion + detector (TarDAL, E2E-MFD fused-image rows) are image-fusion pipelines, not end-to-end detectors, and use YOLOv5s or YOLOv5.
- For a TIP comparison table, either re-run baselines on one declared split, or present only the rows that share a split and state it.
- Fusion-Mamba, WaveMamba, IRDFusion and YOLOv11-RGBT also use COCO-pretrained or large backbones; E2E-MFD's 91.8 is not a pure RGB-T backbone comparison.

## KAIST (log-average miss rate, reasonable subset, lower is better, IoU 0.5, 2,252 test pairs)

All rows are the reasonable setting. "Train annotations" are original or sanitized (Li et al., MSDS-RCNN, or AR-CNN's KAIST-Paired). "Test annotations" are the improved (sanitized) annotations of Liu et al. (BMVC 2016) in every row where the paper says so; I marked rows where it does not.

| Method | Venue | Backbone | Train annotations | Test annotations | All | Day | Night | Source |
|---|---|---|---|---|---|---|---|---|
| AR-CNN | ICCV 2019 | VGG-16 | KAIST-Paired (sanitized) | Liu et al. improved | 9.34 | 9.94 | 8.38 | zhang2019arcnn, Table 2 |
| CIAN | Information Fusion 2019 | VGG-16 | not restated | Liu et al. improved | 14.12 | 14.77 | 11.13 | zhang2021gaff, Table 6a (secondary; MBNet Table 1 prints the same) |
| MBNet | ECCV 2020 | ResNet-50 | AR-CNN improved | Liu et al. improved | 8.13 | 8.28 | 7.86 | zhou2020mbnet, Table 1 (640x512 input) |
| GAFF, sanitized training | WACV 2021 | VGG-16 | sanitized | Liu et al. improved | 6.48 | 8.35 | 3.46 | zhang2021gaff, Table 6b |
| GAFF, original training | WACV 2021 | VGG-16 | original | Liu et al. improved | 10.62 | 10.82 | 10.14 | zhang2021gaff, Table 6a |
| ICAFusion | Pattern Recognition 2024 | row "Ours"; text says YOLOv5 with CSPDarknet53 was best | sanitized (AR-CNN) | Liu et al. improved | 7.17 | 6.82 | 7.85 | shen2024icafusion, Table 9 (input 640x640 train, 640x512 test) |
| MS-DETR | IEEE TITS 2024 | ResNet-50 | sanitized, 7,601 images | Liu et al. improved | 6.13 | 7.78 | 3.18 | xing2024msdetr, Table I (R-All, R-Day, R-Night) |
| TFDet | IEEE TNNLS 2024 | VGG-16, Faster R-CNN | sanitized, 7,601 pairs | updated annotations (MSDS-RCNN and Liu et al.) | 4.47 | 5.22 | 3.36 | zhang2024tfdet, Table III |
| Bag-of-tricks best model | arXiv 2024 (TPAMI special issue draft) | not stated | not stated | not stated | 6.23 | 6.91 | 6.19 | zhou2024tricks, Table 4 |

Other numbers printed in the same papers for context: MS-DETR Table I also lists MLPD 7.58 / 7.96 / 6.95 and GAFF 6.48 / 8.35 / 3.46 under the same protocol, and under the "All" setting (all pedestrians) MS-DETR gets 20.64 / 22.76 / 15.77 (Table I). RSDet reports KAIST only under the "All" setting at IoU 0.5 (24.79 all, 24.18 day, 26.49 night; Table IV, ResNet50), which is not comparable to the reasonable numbers above and is not in the table.

KAIST caveats.
- Original versus sanitized annotation changes the score a lot. GAFF is 10.62 with original training annotations and 6.48 with sanitized ones (Table 6). Always state both train and test annotations.
- The test set is the 2,252-image Liu et al. improved set in nearly every recent paper. The training set is 7,601 sanitized images (MSDS-RCNN, AR-CNN) in most; ICAFusion's text counts 8,963 training pairs, GAFF 7,601, MBNet follows AR-CNN. Note this when comparing.
- "Reasonable" means height of 55 pixels or more with no or partial occlusion. GAFF's footnote says pedestrians taller than 50 pixels; use the 55-pixel definition and the KAISTdevkit-matlab evaluation code from MSDS-RCNN.
- The same method differs by a few hundredths across papers (MBNet 8.13 in its paper and MS-DETR, 8.40 in ICAFusion Table 9, 8.43 in the bag-of-tricks Table 4; CIAN 14.77 versus 14.78 for Day). Quote a method from its own paper.
- Low-miss-rate leaders (TFDet 4.47, MS-DETR 6.13, GAFF 6.48) use VGG-16 or ResNet-50 two-stage or DETR detectors, not YOLO. The ICAFusion KAIST row is a YOLO-style detector at 38.46 Hz on an RTX 3090 (Table 9).

## Reading notes for the paper

- Safest comparison table: LLVIP with CFT, CSSA, MS-DETR, DAMSDet, Fusion-Mamba, TFDet (same official split), reporting input size; KAIST with MBNet, GAFF, MS-DETR, TFDet and AR-CNN (same Liu test annotations); M3FD only against baselines you re-run on one split.
- Papers that were the source of a number but are not themselves RGB-T detectors (LLVIP paper, TarDAL) are dataset or fusion papers; label them as baselines.
- arXiv-only entries (CFT, RSDet at the time of reading, IRDFusion, YOLOv11-RGBT, bag-of-tricks) have not been peer reviewed in the version I read. RSDet's arXiv metadata lists IEEE TITS 2026 as the journal reference; I did not open the journal page.
