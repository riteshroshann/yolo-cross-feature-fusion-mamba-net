"""Phase 2: train the Temporal State Memory on clips, run sequences causally, inject temporal faults, track."""
from __future__ import annotations

import random
import time
from types import SimpleNamespace

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from . import faults, kaist
from ultralytics.utils import nms


def read_frame(visible) -> np.ndarray:
    """(H, W, 4) uint8: KAIST visible BGR + lwir grey."""
    vis = cv2.imread(str(visible), cv2.IMREAD_COLOR)
    ir = cv2.imread(str(kaist.lwir(visible)), cv2.IMREAD_GRAYSCALE)
    if ir.shape != vis.shape[:2]:
        ir = cv2.resize(ir, (vis.shape[1], vis.shape[0]))
    return np.concatenate([vis, ir[..., None]], 2)


def to_tensor(frames: np.ndarray) -> torch.Tensor:
    """(..., H, W, 4) uint8 -> (..., 4, H, W) float in [0, 1]."""
    t = torch.from_numpy(np.ascontiguousarray(frames))
    return t.movedim(-1, -3).float().div_(255)


def periodic(fault="thermal_drop", severity=3, on=40, off=40):
    """Schedule (active, fault, severity): healthy for `off` frames, then failed for `on`, repeating."""
    def active(t):
        return t % (on + off) >= off
    return active, fault, severity


class ClipDataset(Dataset):
    """Clips of T frames at a fixed stride from KAIST training sequences, with labels for every frame."""

    def __init__(self, seqs, T=6, stride=2, fault_p=0.5, flip=True, min_h=20):
        self.T, self.stride, self.fault_p, self.flip, self.min_h = T, stride, fault_p, flip, min_h
        self.items = [(s, i) for s in seqs for i in range(0, len(s.frames) - (T - 1) * stride, stride * T)]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, k):
        seq, i0 = self.items[k]
        idx = [i0 + j * self.stride for j in range(self.T)]
        frames = np.stack([read_frame(seq.frames[i]) for i in idx])
        h, w = frames.shape[1:3]
        flip = self.flip and random.random() < 0.5
        if flip:
            frames = frames[:, :, ::-1]
        labels = []
        for j, i in enumerate(idx):
            for b in seq.boxes[i]:
                if b.label != "person" or b.xywh[3] < self.min_h:
                    continue
                x1, y1, x2, y2 = (max(0, b.xyxy()[0]), max(0, b.xyxy()[1]), min(w, b.xyxy()[2]), min(h, b.xyxy()[3]))
                if x2 - x1 < 2 or y2 - y1 < 2:
                    continue
                cx = (x1 + x2) / 2 / w
                labels.append([j, 0, 1 - cx if flip else cx, (y1 + y2) / 2 / h, (x2 - x1) / w, (y2 - y1) / h])
        fault = None
        if random.random() < self.fault_p:
            kind = random.choice([f for f in faults.FAULTS if f.endswith(("drop", "dark", "noise", "glare", "shift"))])
            start = random.randint(1, self.T - 1)
            fault = (kind, random.randint(2, 3), start, self.T - start)
        return torch.from_numpy(np.ascontiguousarray(frames)), torch.tensor(labels, dtype=torch.float32).view(-1, 6), fault


def collate(batch):
    clips = torch.stack([b[0] for b in batch])
    labels = [torch.cat([torch.full((len(b[1]), 1), float(i)), b[1]], 1) for i, b in enumerate(batch)]
    return clips, torch.cat(labels) if labels else torch.zeros(0, 7), [b[2] for b in batch]


def _apply_clip_faults(x, t, clip_faults):
    for i, f in enumerate(clip_faults):
        if f is not None and f[2] <= t < f[2] + f[3]:
            x[i:i + 1] = faults.apply(x[i:i + 1], f[0], f[1], seed=t)
    return x


def train_memory(model, seqs, *, epochs=4, T=6, stride=2, batch=4, lr=2e-3, workers=4, device="cuda",
                 fault_p=0.5, log=print, amp=True):
    """Train only the memory, detector frozen, by backprop through T frames; non-finite steps are skipped."""
    from ultralytics.cfg import get_cfg
    model.to(device).eval()
    if getattr(model, "memory", None) is None:
        model.attach_memory()
    model.args = get_cfg()
    crit = model.init_criterion()
    for p in model.parameters():
        p.requires_grad_(False)
    for p in model.memory.parameters():
        p.requires_grad_(True)
    model.memory.train()
    opt = torch.optim.AdamW(model.memory.parameters(), lr=lr, weight_decay=1e-4)
    loader = DataLoader(ClipDataset(seqs, T, stride, fault_p), batch_size=batch, shuffle=True,
                        num_workers=workers, collate_fn=collate, drop_last=True, persistent_workers=workers > 0)
    steps = epochs * len(loader)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=steps, pct_start=0.1)
    scaler = torch.amp.GradScaler(enabled=amp)
    history, step, skipped = [], 0, 0
    for ep in range(epochs):
        t0, run = time.time(), 0.0
        for clips, labels, clip_faults in loader:
            clips = clips.to(device, non_blocking=True)
            labels = labels.to(device)
            model.reset_memory()
            total = 0.0
            with torch.autocast("cuda", torch.float16, enabled=amp):
                for t in range(T):
                    x = clips[:, t].permute(0, 3, 1, 2).float().div(255)
                    x = _apply_clip_faults(x, t, clip_faults)
                    preds = model(x)
                    sel = labels[:, 1] == t
                    lt = labels[sel]
                    b = {"batch_idx": lt[:, 0], "cls": lt[:, 2:3], "bboxes": lt[:, 3:7], "img": x}
                    loss, _ = crit(preds, b)
                    total = total + loss.sum() / T
            opt.zero_grad(set_to_none=True)
            if not torch.isfinite(total):
                skipped += 1
                if skipped > max(20, steps // 20):
                    raise FloatingPointError(f"memory loss non-finite on {skipped} of {step + 1} steps")
                step += 1
                continue
            scaler.scale(total).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.memory.parameters(), 10.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            run += total.item()
            step += 1
            if step % 50 == 0:
                log(f"epoch {ep + 1}/{epochs} step {step}/{steps} loss {run / 50:.3f}")
                history.append((step, run / 50))
                run = 0.0
        log(f"epoch {ep + 1} done in {(time.time() - t0) / 60:.1f} min, {skipped} non-finite steps skipped")
    model.memory.eval()
    return history


def detect(model, x, conf=0.001, iou=0.6, max_det=300):
    """List of (n, 6) numpy arrays [x1 y1 x2 y2 conf cls] for a (b, 4, H, W) batch at input resolution."""
    out = model(x)
    out = out[0] if isinstance(out, (tuple, list)) else out
    dets = nms.non_max_suppression(out.float(), conf, iou, max_det=max_det, end2end=getattr(model, "end2end", False))
    return [d.cpu().numpy() for d in dets]


@torch.no_grad()
def run_sequence(model, seq, *, stride=2, memory=True, schedule=None, eval_every=20, device="cuda", conf=0.001,
                 flags=False, half=True):
    """Run causally at `stride`; return {frame_index: dets} for frames on the eval grid, and fault mask."""
    model.eval()
    has_mem = getattr(model, "memory", None) is not None
    if has_mem:
        model.memory_on = memory
        model.reset_memory()
    out, failed = {}, {}
    active, fault, sev = schedule if schedule else (lambda t: False, None, 0)
    for i in range(0, len(seq.frames), stride):
        x = to_tensor(read_frame(seq.frames[i]))[None].to(device)
        bad = bool(active(i))
        if bad:
            x = faults.apply(x, fault, sev, seed=i)
        model.sensor_flags = torch.tensor(faults.flags_for(fault), device=device) if (bad and flags and faults.flags_for(fault)) else None
        with torch.autocast("cuda", torch.float16, enabled=half and x.is_cuda):
            d = detect(model, x, conf=conf)[0]
        if i % eval_every == 0:
            out[i] = d
            failed[i] = bad
    model.sensor_flags = None
    return out, failed


def evaluate_sequences(model, seqs, **kw) -> dict:
    """MR^-2 (reasonable, all / day / night / failed frames) and AP50 over the eval grid of every sequence."""
    rows = {"all": ([], []), "day": ([], []), "night": ([], []), "failed": ([], []), "healthy": ([], [])}
    for s in seqs:
        dets, failed = run_sequence(model, s, **kw)
        for i, d in dets.items():
            for key in ("all", "day" if s.daytime else "night", "failed" if failed[i] else "healthy"):
                rows[key][0].append(d)
                rows[key][1].append(s.boxes[i])
    res = {}
    for k, (d, b) in rows.items():
        if d:
            res[f"MR_{k}"] = kaist.miss_rate(d, b)
            res[f"AP50_{k}"] = kaist.average_precision(d, b)
            res[f"n_{k}"] = len(d)
    return res


class _Dets:
    def __init__(self, d):
        self.d = d
        self.xyxy = d[:, :4]
        self.conf = d[:, 4]
        self.cls = d[:, 5]
        self.xywh = np.stack([(d[:, 0] + d[:, 2]) / 2, (d[:, 1] + d[:, 3]) / 2, d[:, 2] - d[:, 0], d[:, 3] - d[:, 1]], 1)

    def __len__(self):
        return len(self.d)

    def __getitem__(self, k):
        return _Dets(self.d[k])


def tracker(**over):
    from ultralytics.trackers.byte_tracker import BYTETracker
    from ultralytics.utils import YAML
    from ultralytics.utils.checks import check_yaml
    cfg = {**YAML.load(check_yaml("bytetrack.yaml")), **over}
    return BYTETracker(SimpleNamespace(**cfg))


@torch.no_grad()
def track_sequence(model, seq, *, stride=1, memory=True, schedule=None, device="cuda", conf=0.1, flags=False):
    """ByteTrack over every `stride`-th frame; returns {frame: (n, 5) [x1 y1 x2 y2 id]}."""
    model.eval()
    if getattr(model, "memory", None) is not None:
        model.memory_on = memory
        model.reset_memory()
    trk = tracker()
    active, fault, sev = schedule if schedule else (lambda t: False, None, 0)
    out = {}
    for i in range(0, len(seq.frames), stride):
        x = to_tensor(read_frame(seq.frames[i]))[None].to(device)
        bad = bool(active(i))
        if bad:
            x = faults.apply(x, fault, sev, seed=i)
        model.sensor_flags = torch.tensor(faults.flags_for(fault), device=device) if (bad and flags and faults.flags_for(fault)) else None
        with torch.autocast("cuda", torch.float16, enabled=x.is_cuda):
            d = detect(model, x, conf=conf)[0]
        tr = trk.update(_Dets(d.reshape(-1, 6)))
        out[i] = np.asarray(tr)[:, :5] if len(tr) else np.zeros((0, 5))
    model.sensor_flags = None
    return out


def mot_metrics(tracks: dict, seq, min_h=30) -> dict:
    """MOTA, IDF1, ID switches against KAIST person tracks; predictions on ignore regions are dropped."""
    import motmetrics as mm
    acc = mm.MOTAccumulator(auto_id=True)
    for i, tr in tracks.items():
        boxes = seq.boxes[i]
        gt = [b for b in boxes if b.label == "person" and b.xywh[3] >= min_h]
        ign = [b for b in boxes if not (b.label == "person" and b.xywh[3] >= min_h)]
        keep = []
        for t in tr:
            hit_ignore = any(_iou(t[:4], b.xyxy()) > 0.5 for b in ign)
            if not hit_ignore:
                keep.append(t)
        keep = np.asarray(keep).reshape(-1, 5)
        g = np.array([b.xywh for b in gt]).reshape(-1, 4)
        h = np.stack([keep[:, 0], keep[:, 1], keep[:, 2] - keep[:, 0], keep[:, 3] - keep[:, 1]], 1) if len(keep) else np.zeros((0, 4))
        dist = mm.distances.iou_matrix(g, h, max_iou=0.5)
        acc.update([b.tid for b in gt], keep[:, 4].astype(int).tolist(), dist)
    s = mm.metrics.create().compute(acc, metrics=["mota", "idf1", "num_switches", "mostly_tracked", "num_unique_objects"])
    return {k: float(v) for k, v in s.iloc[0].items()}


def _iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / max((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter, 1e-9)
