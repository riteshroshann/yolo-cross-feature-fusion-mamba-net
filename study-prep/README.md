# Study prep

Everything needed to understand CFFM-Net from first principles, in the order it is needed. Read
`CFFM-Net_Study_Guide.pdf` for the full path: what to learn in each stage, why it matters for this project,
which files in the repository to read afterwards, and exercises.

## Stage 1 · Foundations (3 to 4 weeks)

| resource | link |
|---|---|
| Stanford CS231n, course site and notes | https://cs231n.stanford.edu/ · https://cs231n.github.io/ |
| CS231n lectures (Spring 2017) | https://www.youtube.com/playlist?list=PL3FW7Lu3i5JvHM8ljYj-zLfQRF3EO8sYv |
| Andrej Karpathy, Neural Networks: Zero to Hero | https://karpathy.ai/zero-to-hero.html · https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ |
| Michigan EECS 498, Deep Learning for Computer Vision (Justin Johnson) | https://www.youtube.com/playlist?list=PL5-TkQAfAZFbzxjBHtzdVCWE0Zbhomg7r |
| 3Blue1Brown, neural networks | https://www.3blue1brown.com/topics/neural-networks |
| Dive into Deep Learning (book, free) | https://d2l.ai/ |
| MIT 6.S191, Introduction to Deep Learning | https://introtodeeplearning.com/ |

Papers in `papers/1_foundations`: ResNet, Batch Normalization, Adam, Attention Is All You Need, ViT.

## Stage 2 · Detection and YOLO (2 weeks)

| resource | link |
|---|---|
| Ultralytics documentation | https://docs.ultralytics.com/ |
| YOLO26, the detector this project builds on | https://docs.ultralytics.com/models/yolo26/ |
| The Illustrated Transformer (for DETR) | https://jalammar.github.io/illustrated-transformer/ |
| The Annotated Transformer | https://nlp.seas.harvard.edu/annotated-transformer/ |

Papers in `papers/2_detection_and_yolo`: Faster R-CNN, YOLOv1, FPN, RetinaNet, YOLOv3, DETR, RT-DETR, YOLOv10.

## Stage 3 · State-space models and Mamba (2 weeks)

| resource | link |
|---|---|
| The Annotated S4 (Sasha Rush) | https://srush.github.io/annotated-s4/ |
| A Visual Guide to Mamba (Maarten Grootendorst) | https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-mamba-and-state |
| Official Mamba code | https://github.com/state-spaces/mamba |
| Stanford CS25, Transformers United | https://web.stanford.edu/class/cs25/ |

Papers in `papers/3_state_space_models_mamba`: S4, Mamba, Mamba-2, Vision Mamba, VMamba, MambaOut, MambaVision,
Mamba YOLO.

## Stage 4 · Visible and thermal fusion (1 week)

Papers in `papers/4_multispectral_fusion`: CFT, ICAFusion, Fusion-Mamba, CFMW, COMO, and the datasets KAIST,
LLVIP, M3FD and RGBT-Tiny. The full set of 150 papers behind the project is in `../papers/`.

## Stage 5 · Tracking and video (1 week)

| resource | link |
|---|---|
| Ultralytics tracking (ByteTrack, BoT-SORT) | https://docs.ultralytics.com/modes/track/ |
| ByteTrack code | https://github.com/ifzhang/ByteTrack |

Papers in `papers/5_tracking_and_video`: SORT, ByteTrack, OC-SORT, BoT-SORT, MOT16/17, VT-MOT, MambaST.

## Reading the code after each stage

| after | read |
|---|---|
| stage 1 | `cffm-net-pilot/src/cffm/scan.py` |
| stage 2 | `cffm-net-pilot/src/cffm/model.py`, `train.py` |
| stage 3 | `cffm-net-pilot/src/cffm/blocks.py` |
| stage 4 | `cffm-net-pilot/src/cffm/data.py`, notebook 06 |
| stage 5 | `cffm-net-pilot/src/cffm/hud.py`, `showcase.py`, notebook 16 |
