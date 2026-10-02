"""CFFM-Net pilot: Cross-Feature Fusion Mamba Network for visible-thermal small-object detection.

Modules
-------
scan    selective scan, CUDA kernel or portable chunked PyTorch scan
blocks  ReliabilityHead, OffsetAlign, GatedCrossScan, CMFM and the baseline fusers
model   DualStreamDetectionModel: two YOLO26 backbones + fusion + YOLO26 neck/head
data    paired visible-thermal datasets for Ultralytics, LLVIP and M3FD converters
train   DualTrainer / DualValidator, train_run, evaluate, probe, latency
tsm     Temporal State Memory (Phase 2, tested but not used in the pilot)
env     platform detection, paths, GPU report
viz     drawing predictions and reliability maps
"""
__version__ = "0.1.0"
