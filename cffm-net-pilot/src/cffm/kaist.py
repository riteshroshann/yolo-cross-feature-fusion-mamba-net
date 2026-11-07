"""KAIST multispectral pedestrian benchmark: vbb annotations, paired layout, sequences and log-average miss rate."""
from __future__ import annotations

import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

TRAIN_SETS = tuple(f"set{i:02d}" for i in range(0, 6))
TEST_SETS = tuple(f"set{i:02d}" for i in range(6, 12))
DAY_SETS = {"set00", "set01", "set02", "set06", "set07", "set08"}
REASONABLE_MIN_H = 55
REASONABLE_MIN_VIS = 0.65
FPS = 20


@dataclass
class Box:
    tid: int
    label: str
    xywh: tuple
    vis: float

    @property
    def reasonable(self) -> bool:
        return self.label == "person" and self.xywh[3] >= REASONABLE_MIN_H and self.vis >= REASONABLE_MIN_VIS

    def xyxy(self):
        x, y, w, h = self.xywh
        return x, y, x + w, y + h


@dataclass
class Sequence:
    name: str
    frames: list = field(default_factory=list)
    boxes: list = field(default_factory=list)

    @property
    def daytime(self) -> bool:
        return self.name.split("/")[0] in DAY_SETS


def read_vbb(path: str | Path) -> list[list[Box]]:
    """Per-frame boxes from a Piotr Dollar vbb file."""
    from scipy.io import loadmat
    A = loadmat(str(path))["A"][0][0]
    labels = [str(x[0]) if x.size else "" for x in A["objLbl"][0]]
    out = []
    for objs in A["objLists"][0]:
        frame = []
        if objs.size:
            for o in objs[0]:
                tid = int(o["id"].item()) - 1
                x, y, w, h = (float(v) for v in o["pos"].ravel())
                vis = 1.0
                if int(o["occl"].item()):
                    pv = o["posv"].ravel().astype(float)
                    vis = float(pv[2] * pv[3] / max(w * h, 1e-6)) if pv.size == 4 and pv[2] > 0 else 0.0
                frame.append(Box(tid, labels[tid] if 0 <= tid < len(labels) else "person", (x, y, w, h), vis))
        out.append(frame)
    return out


def find_roots(roots) -> tuple[Path, Path]:
    """(image root holding setXX/VYYY/visible, annotation root holding setXX/VYYY.vbb)."""
    img = ann = None
    for r in map(Path, roots):
        if not r.exists():
            continue
        for p in r.rglob("V000.vbb"):
            ann = ann or p.parent.parent
        for p in r.glob("**/set00/V000/visible"):
            img = img or p.parent.parent.parent
        if img and ann:
            break
    if img is None or ann is None:
        raise FileNotFoundError("KAIST images (setXX/VYYY/visible) or annotations (setXX/VYYY.vbb) not found")
    return img, ann


def sequences(img_root, ann_root, sets=TEST_SETS) -> list[Sequence]:
    """Every annotated sequence in `sets`, frames that exist on disk only."""
    out = []
    for s in sets:
        for vbb in sorted(Path(ann_root, s).glob("V*.vbb")):
            name = f"{s}/{vbb.stem}"
            vis_dir = Path(img_root, s, vbb.stem, "visible")
            if not vis_dir.exists():
                continue
            boxes = read_vbb(vbb)
            seq = Sequence(name)
            for i, b in enumerate(boxes):
                f = vis_dir / f"I{i:05d}.jpg"
                if f.exists():
                    seq.frames.append(f)
                    seq.boxes.append(b)
            out.append(seq)
    return out


def lwir(visible: Path) -> Path:
    return visible.parent.parent / "lwir" / visible.name


def _yolo_lines(boxes, w=640, h=512, min_h=20):
    lines = []
    for b in boxes:
        if b.label != "person" or b.xywh[3] < min_h:
            continue
        x1, y1, x2, y2 = b.xyxy()
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
        if x2 - x1 < 2 or y2 - y1 < 2:
            continue
        lines.append(f"0 {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}")
    return lines


def _place(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.symlink(src, dst)
    except OSError:
        shutil.copyfile(src, dst)


def convert(img_root, ann_root, out, train_stride=2, test_stride=20, workers=16) -> dict:
    """KAIST -> paired YOLO layout: train every 2nd frame of set00-05, val the standard test-20 of set06-11."""
    from .data import write_yamls
    out = Path(out)
    if out.exists():
        shutil.rmtree(out)
    jobs = []
    for split, sets, stride in (("train", TRAIN_SETS, train_stride), ("val", TEST_SETS, test_stride)):
        for seq in sequences(img_root, ann_root, sets):
            for f, b in list(zip(seq.frames, seq.boxes))[::stride]:
                stem = f"{seq.name.replace('/', '_')}_{f.stem}"
                jobs.append((split, stem, f, b))
    counts = {s: sum(1 for j in jobs if j[0] == s) for s in ("train", "val")}
    if not all(counts.values()):
        raise FileNotFoundError(f"KAIST split is empty {counts}; the image dataset needs set00-05 and set06-11")

    def write(job):
        split, stem, f, b = job
        _place(f, out / "images" / "visible" / split / f"{stem}.jpg")
        _place(lwir(f), out / "images" / "infrared" / split / f"{stem}.jpg")
        text = "\n".join(_yolo_lines(b))
        for mod in ("visible", "infrared"):
            p = out / "labels" / mod / split / f"{stem}.txt"
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(text + ("\n" if text else ""))

    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(write, jobs))
    write_yamls(out, ["person"])
    return counts


def _match(dets, gts, ignore, iou_thr=0.5):
    """Greedy Caltech matching: tp/fp per detection (sorted by score), ignore-matched detections dropped."""
    if len(dets) == 0:
        return np.zeros(0), np.zeros(0), np.zeros(0)
    order = np.argsort(-dets[:, 4])
    dets = dets[order]
    gts = np.asarray(gts, float).reshape(-1, 4)
    ignore = np.asarray(ignore, bool)
    gts_order = np.argsort(ignore)
    gts, ignore = gts[gts_order], ignore[gts_order]
    used = np.zeros(len(gts), bool)
    tp, fp, keep = np.zeros(len(dets)), np.zeros(len(dets)), np.ones(len(dets), bool)
    for i, d in enumerate(dets):
        best, bj = iou_thr, -1
        for j, g in enumerate(gts):
            if used[j] and not ignore[j]:
                continue
            if bj >= 0 and not ignore[bj] and ignore[j]:
                break
            ix = max(0.0, min(d[2], g[2]) - max(d[0], g[0]))
            iy = max(0.0, min(d[3], g[3]) - max(d[1], g[1]))
            inter = ix * iy
            union = inter if ignore[j] else (d[2] - d[0]) * (d[3] - d[1]) + (g[2] - g[0]) * (g[3] - g[1]) - inter
            o = inter / max(union, 1e-9)
            if o >= best:
                best, bj = o, j
        if bj < 0:
            fp[i] = 1
        elif ignore[bj]:
            keep[i] = False
        else:
            used[bj] = True
            tp[i] = 1
    return dets[keep, 4], tp[keep], fp[keep]


def miss_rate(frames_dets, frames_boxes, subset="reasonable") -> float:
    """Log-average miss rate over FPPI in [1e-2, 1e0] (Caltech protocol); lower is better."""
    scores, tps, fps, n_gt = [], [], [], 0
    min_h = REASONABLE_MIN_H if subset == "reasonable" else 0
    for dets, boxes in zip(frames_dets, frames_boxes):
        gts = [b.xyxy() for b in boxes]
        ign = [not (b.reasonable if subset == "reasonable" else b.label == "person") for b in boxes]
        n_gt += sum(1 for i in ign if not i)
        d = np.asarray(dets, float).reshape(-1, 6)
        d = d[(d[:, 3] - d[:, 1]) >= min_h / 1.25]
        s, tp, fp = _match(d[:, :5], gts, ign)
        scores.append(s)
        tps.append(tp)
        fps.append(fp)
    s = np.concatenate(scores) if scores else np.zeros(0)
    order = np.argsort(-s)
    tp = np.cumsum(np.concatenate(tps)[order]) if len(s) else np.zeros(1)
    fp = np.cumsum(np.concatenate(fps)[order]) if len(s) else np.zeros(1)
    mr = 1 - tp / max(n_gt, 1)
    fppi = fp / max(len(frames_dets), 1)
    refs = np.logspace(-2, 0, 9)
    vals = []
    for r in refs:
        idx = np.where(fppi <= r)[0]
        vals.append(mr[idx[-1]] if len(idx) else 1.0)
    return float(np.exp(np.mean(np.log(np.maximum(vals, 1e-10)))))


def average_precision(frames_dets, frames_boxes, subset="reasonable") -> float:
    """All-point interpolated AP at IoU 0.5 with the same ignore rules as the miss rate."""
    scores, tps, n_gt = [], [], 0
    for dets, boxes in zip(frames_dets, frames_boxes):
        ign = [not (b.reasonable if subset == "reasonable" else b.label == "person") for b in boxes]
        n_gt += sum(1 for i in ign if not i)
        d = np.asarray(dets, float).reshape(-1, 6)
        s, tp, _ = _match(d[:, :5], [b.xyxy() for b in boxes], ign)
        scores.append(s)
        tps.append(tp)
    if not n_gt:
        return float("nan")
    s = np.concatenate(scores)
    tp = np.concatenate(tps)[np.argsort(-s)]
    ctp = np.cumsum(tp)
    rec = ctp / n_gt
    prec = ctp / np.arange(1, len(tp) + 1)
    mrec = np.concatenate([[0], rec, [1]])
    mpre = np.concatenate([[1], prec, [0]])
    mpre = np.maximum.accumulate(mpre[::-1])[::-1]
    i = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1]))
