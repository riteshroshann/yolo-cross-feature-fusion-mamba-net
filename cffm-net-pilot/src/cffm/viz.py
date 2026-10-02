"""
Looking at what the model does.

Two questions we always want answered with pictures, not just numbers:
  1. What did it detect, and what did it miss?           -> predict_pair, draw_pair
  2. Where did it trust visible, and where thermal?      -> reliability_maps, plot_reliability

The reliability maps are the most direct evidence that gating works: in a night
scene the thermal map should light up where people are and the visible map
should fade. If it doesn't, the numbers are lying to us somewhere.
"""
from __future__ import annotations

import cv2
import numpy as np
import torch
from ultralytics.data.augment import LetterBox
from ultralytics.utils import nms
from ultralytics.utils.ops import scale_boxes

from .data import read_pair


def load_pair(visible_path, imgsz=640, device="cpu"):
    """Read a pair, letterbox it like the validator does, return (tensor, pair_image)."""
    pair = read_pair(visible_path)                                 # (H, W, 4) uint8
    if pair is None:
        raise FileNotFoundError(visible_path)
    lb = LetterBox(new_shape=(imgsz, imgsz), auto=False, scaleup=False)(image=pair)
    if lb.ndim == 2 or lb.shape[2] != 4:                           # cv2 may drop the 4th channel on padding
        raise RuntimeError("letterbox lost the thermal channel")
    x = torch.from_numpy(np.ascontiguousarray(lb.transpose(2, 0, 1))).float().div(255).unsqueeze(0)
    return x.to(device), pair


@torch.no_grad()
def predict_pair(model, visible_path, conf=0.25, iou=0.7, imgsz=640):
    """Boxes (n, 6: x1 y1 x2 y2 conf cls) in original pixel coordinates."""
    device = next(model.parameters()).device
    x, pair = load_pair(visible_path, imgsz, device)
    x = x.to(next(model.parameters()).dtype)
    preds = model(x)
    preds = preds[0] if isinstance(preds, (tuple, list)) else preds
    det = nms.non_max_suppression(preds, conf, iou, end2end=getattr(model, "end2end", False))[0]
    det[:, :4] = scale_boxes(x.shape[2:], det[:, :4], pair.shape[:2])
    return det.cpu().numpy(), pair


def draw_pair(pair, det, names, gt=None):
    """Visible and thermal side by side (RGB), predictions in green, ground truth in red."""
    vis = np.ascontiguousarray(pair[..., :3])
    ir = cv2.cvtColor(np.ascontiguousarray(pair[..., 3]), cv2.COLOR_GRAY2BGR)
    for im in (vis, ir):
        for b in (gt if gt is not None else []):
            x1, y1, x2, y2 = map(int, b[:4])
            cv2.rectangle(im, (x1, y1), (x2, y2), (40, 40, 220), 1)
        for x1, y1, x2, y2, c, k in det:
            cv2.rectangle(im, (int(x1), int(y1)), (int(x2), int(y2)), (60, 200, 60), 2)
            cv2.putText(im, f"{names[int(k)]} {c:.2f}", (int(x1), max(10, int(y1) - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (60, 200, 60), 1, cv2.LINE_AA)
    both = np.concatenate([vis, ir], axis=1)
    return cv2.cvtColor(both, cv2.COLOR_BGR2RGB)


@torch.no_grad()
def reliability_maps(model, visible_path, imgsz=640):
    """Run one pair through the model and collect r_v, r_t at every fused level.

    Returns {level: (r_visible, r_thermal)} as float arrays at the letterboxed
    image size, plus the letterboxed pair for plotting underneath.
    """
    device = next(model.parameters()).device
    x, _ = load_pair(visible_path, imgsz, device)
    model(x.to(next(model.parameters()).dtype))
    out = {}
    for i, f in model.fusers.items():
        last = getattr(f, "last", {})
        if "rv" in last:
            up = lambda r: torch.nn.functional.interpolate(r.float(), size=x.shape[2:], mode="bilinear",
                                                           align_corners=False)[0, 0].cpu().numpy()
            out[model.levels[int(i)]] = (up(last["rv"]), up(last["rt"]))
    lb = (x[0].cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
    return out, lb


def plot_reliability(maps, lb, level=3, ax=None):
    """visible | thermal | r_visible | r_thermal at one level, on four matplotlib axes."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(1, 4, figsize=(16, 3.6))
    rv, rt = maps[level]
    ax[0].imshow(cv2.cvtColor(lb[..., :3], cv2.COLOR_BGR2RGB)); ax[0].set_title("visible")
    ax[1].imshow(lb[..., 3], cmap="gray"); ax[1].set_title("thermal")
    ax[2].imshow(rv, vmin=0, vmax=1, cmap="magma"); ax[2].set_title(f"r_visible at P{level}")
    ax[3].imshow(rt, vmin=0, vmax=1, cmap="magma"); ax[3].set_title(f"r_thermal at P{level}")
    for a in ax:
        a.axis("off")
    return ax
