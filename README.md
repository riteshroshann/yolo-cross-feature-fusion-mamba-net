# cffm-net

Object detection from a visible and a thermal camera at once. Two YOLO26 backbones, fused by a Mamba-style
selective scan. Each sensor's reliability, at every location, scales the scan's step size, so a blinded
camera cannot overwrite what the other one sees. The focus is small objects.

Status: the pilot (still images, LLVIP and M3FD) is written and tested, but not trained yet. Video, live
cameras and multiple cameras come after it.

```
cffm-net-pilot/   code and notebooks for the pilot, runs on Kaggle (2x T4)
docs/dossier/     research dossier: survey, method, plan
docs/novelty/     what the pilot adds, and the limitations of prior work it targets
docs/datasets/    which datasets, and how to get them
papers/           the open-access papers the dossier cites
```

To run the pilot, see `cffm-net-pilot/README.md`.
