"""Phase 4: calibrated multi-camera fusion on a ground plane, anonymous tracking and aggregate analytics."""
from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

CAMS = ("CVLab1", "CVLab2", "CVLab3", "CVLab4", "IDIAP1", "IDIAP2", "IDIAP3")
GRID = (480, 1440)
CELL_CM = 2.5
ORIGIN_CM = (-300.0, -900.0)
FRAMES = tuple(range(0, 2000, 5))
TEST_FROM = 1800


@dataclass
class View:
    name: str
    K: np.ndarray
    R: np.ndarray
    t: np.ndarray

    @property
    def H(self) -> np.ndarray:
        """Ground (x, y, 1) in cm -> image pixels."""
        return self.K @ np.column_stack([self.R[:, 0], self.R[:, 1], self.t])

    def to_image(self, xy: np.ndarray) -> np.ndarray:
        p = np.column_stack([xy, np.ones(len(xy))]) @ self.H.T
        return p[:, :2] / p[:, 2:3]

    def to_ground(self, uv: np.ndarray) -> np.ndarray:
        p = np.column_stack([uv, np.ones(len(uv))]) @ np.linalg.inv(self.H).T
        return p[:, :2] / p[:, 2:3]

    def faces_ground(self, uv: np.ndarray) -> np.ndarray:
        """True where the back-projected ray hits the ground in front of the camera."""
        p = np.column_stack([uv, np.ones(len(uv))]) @ np.linalg.inv(self.H).T
        return p[:, 2] > 0


def _numbers(node):
    return np.array([float(v) for v in node.text.split()])


def load_views(root) -> list[View]:
    root = Path(root)
    views = []
    for c in CAMS:
        intr = ET.parse(root / "calibrations" / "intrinsic_zero" / f"intr_{c}.xml").getroot()
        K = _numbers(intr.find("camera_matrix").find("data")).reshape(3, 3)
        extr = ET.parse(root / "calibrations" / "extrinsic" / f"extr_{c}.xml").getroot()
        R, _ = cv2.Rodrigues(_numbers(extr.find("rvec")))
        views.append(View(c, K, R, _numbers(extr.find("tvec"))))
    return views


def position_to_cm(pos) -> np.ndarray:
    pos = np.asarray(pos)
    return np.stack([ORIGIN_CM[0] + CELL_CM * (pos % GRID[0]), ORIGIN_CM[1] + CELL_CM * (pos // GRID[0])], -1)


def in_area(xy: np.ndarray) -> np.ndarray:
    x0, y0 = ORIGIN_CM
    return (xy[:, 0] >= x0) & (xy[:, 0] < x0 + GRID[0] * CELL_CM) & (xy[:, 1] >= y0) & (xy[:, 1] < y0 + GRID[1] * CELL_CM)


def load_annotations(root, frame) -> list[dict]:
    return json.loads((Path(root) / "annotations_positions" / f"{frame:08d}.json").read_text())


def gt_positions(root, frame) -> np.ndarray:
    return position_to_cm([p["positionID"] for p in load_annotations(root, frame)]).reshape(-1, 2)


def image_path(root, cam: int, frame: int) -> Path:
    return Path(root) / "Image_subsets" / f"C{cam + 1}" / f"{frame:08d}.png"


def anonymize(img: np.ndarray, boxes: np.ndarray, head=0.22) -> np.ndarray:
    """Blur the head region of every person box before anything is stored or shown."""
    out = img.copy()
    h, w = img.shape[:2]
    for x1, y1, x2, y2 in np.asarray(boxes, float).reshape(-1, 4)[:, :4]:
        x1, x2 = int(max(0, x1)), int(min(w, x2))
        y1, y2 = int(max(0, y1)), int(min(h, y1 + head * (y2 - y1) + 1))
        if x2 - x1 > 2 and y2 - y1 > 2:
            k = max(3, ((x2 - x1) // 3) | 1)
            out[y1:y2, x1:x2] = cv2.GaussianBlur(out[y1:y2, x1:x2], (k, k), 0)
    return out


def foot_points(view: View, dets: np.ndarray, min_conf=0.0):
    """Ground positions (cm) of the bottom-centre of each person box that lands inside the area."""
    d = np.asarray(dets, float).reshape(-1, 6)
    d = d[(d[:, 5] == 0) & (d[:, 4] >= min_conf)]
    if not len(d):
        return np.zeros((0, 2)), np.zeros(0)
    uv = np.column_stack([(d[:, 0] + d[:, 2]) / 2, d[:, 3]])
    xy = view.to_ground(uv)
    ok = view.faces_ground(uv) & in_area(xy)
    return xy[ok], d[ok, 4]


def fuse(points: list, scores: list, radius=50.0, min_views=1, min_score=0.0):
    """Merge per-view ground points: greedy clusters by score, noisy-OR confidence, score-weighted centre."""
    if not any(len(p) for p in points):
        return np.zeros((0, 2)), np.zeros(0)
    P = np.concatenate([p for p in points if len(p)])
    S = np.concatenate([s for s in scores if len(s)])
    V = np.concatenate([np.full(len(p), i) for i, p in enumerate(points) if len(p)])
    order = np.argsort(-S)
    used = np.zeros(len(P), bool)
    out, conf = [], []
    for i in order:
        if used[i]:
            continue
        near = (~used) & (np.linalg.norm(P - P[i], axis=1) <= radius)
        members = np.where(near)[0]
        best = {}
        for j in members:
            if V[j] not in best or S[j] > S[best[V[j]]]:
                best[V[j]] = j
        sel = np.array(list(best.values()))
        used[members] = True
        c = 1 - np.prod(1 - S[sel])
        if len(sel) >= min_views and c >= min_score:
            out.append((P[sel] * S[sel, None]).sum(0) / S[sel].sum())
            conf.append(c)
    return np.array(out).reshape(-1, 2), np.array(conf)


def visible_in(views, xy, size=(1920, 1080), height_cm=170.0) -> np.ndarray:
    """(n_points, n_views) bool: a person standing at xy would be fully in that camera's frame."""
    xy = np.asarray(xy, float).reshape(-1, 2)
    out = np.zeros((len(xy), len(views)), bool)
    for k, v in enumerate(views):
        foot = np.column_stack([xy, np.zeros(len(xy))])
        head = np.column_stack([xy, np.full(len(xy), -height_cm)])
        ok = np.ones(len(xy), bool)
        for P in (foot, head):
            cam = P @ v.R.T + v.t
            uv = cam @ v.K.T
            uv = uv[:, :2] / np.maximum(uv[:, 2:3], 1e-6)
            ok &= (cam[:, 2] > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < size[0]) & (uv[:, 1] >= 0) & (uv[:, 1] < size[1])
        out[:, k] = ok
    return out


def vote(views, points, scores, radius=50.0, min_ratio=0.5, min_views=2, min_score=0.3, nms_cm=60.0):
    """Visibility-normalised voting: a candidate must be seen by enough of the cameras that could see it."""
    if not any(len(p) for p in points):
        return np.zeros((0, 2)), np.zeros(0)
    cand = np.concatenate([p for p in points if len(p)])
    vis = visible_in(views, cand)
    support = np.zeros((len(cand), len(views)))
    centre = np.zeros((len(cand), 2))
    for k, (p, s) in enumerate(zip(points, scores)):
        if not len(p):
            continue
        D = np.linalg.norm(cand[:, None] - p[None], axis=2)
        near = D <= radius
        best = np.where(near, s[None], 0).max(1)
        support[:, k] = best
    for i in range(len(cand)):
        w, acc = 0.0, np.zeros(2)
        for k, (p, s) in enumerate(zip(points, scores)):
            if not len(p):
                continue
            d = np.linalg.norm(p - cand[i], axis=1)
            j = np.argmin(d)
            if d[j] <= radius:
                acc += s[j] * p[j]
                w += s[j]
        centre[i] = acc / max(w, 1e-9)
    n_vis = np.maximum(vis.sum(1), 1)
    hits = (support > 0)
    ratio = (hits & vis).sum(1) / n_vis
    conf = (support * vis).sum(1) / n_vis
    ok = (hits.sum(1) >= np.minimum(min_views, n_vis)) & (ratio >= min_ratio) & (conf >= min_score)
    cand, conf = centre[ok], conf[ok]
    keep = []
    for i in np.argsort(-conf):
        if all(np.linalg.norm(cand[i] - cand[j]) > nms_cm for j in keep):
            keep.append(i)
    return cand[keep].reshape(-1, 2), conf[keep]


def moda(pred_frames, gt_frames, dist=50.0) -> dict:
    """MODA, MODP, precision and recall on the ground plane with a `dist` cm match radius (WILDTRACK protocol)."""
    tp = fp = fn = 0
    dsum = 0.0
    for pred, gt in zip(pred_frames, gt_frames):
        pred, gt = np.asarray(pred).reshape(-1, 2), np.asarray(gt).reshape(-1, 2)
        if len(pred) and len(gt):
            C = np.linalg.norm(gt[:, None] - pred[None], axis=2)
            r, c = linear_sum_assignment(C)
            ok = C[r, c] <= dist
            m = int(ok.sum())
            dsum += float((1 - C[r, c][ok] / dist).sum())
        else:
            m = 0
        tp += m
        fp += len(pred) - m
        fn += len(gt) - m
    n = tp + fn
    return {"MODA": 1 - (fp + fn) / max(n, 1), "MODP": dsum / max(tp, 1), "precision": tp / max(tp + fp, 1),
            "recall": tp / max(n, 1), "TP": tp, "FP": fp, "FN": fn}


@dataclass
class Track:
    tid: int
    xy: np.ndarray
    v: np.ndarray
    born: int
    last: int
    hits: int = 1
    path: list = field(default_factory=list)


class GroundTracker:
    """Constant-velocity tracks on the ground plane with Hungarian association; ids are ephemeral and anonymous."""

    def __init__(self, gate=75.0, max_miss=3, min_hits=2, alpha=0.6):
        self.gate, self.max_miss, self.min_hits, self.alpha = gate, max_miss, min_hits, alpha
        self.tracks, self.next_id, self.finished = [], 0, []

    def update(self, xy: np.ndarray, t: int) -> list[Track]:
        xy = np.asarray(xy, float).reshape(-1, 2)
        pred = np.array([tr.xy + tr.v * (t - tr.last) for tr in self.tracks]).reshape(-1, 2)
        matched_t, matched_d = set(), set()
        if len(pred) and len(xy):
            C = np.linalg.norm(pred[:, None] - xy[None], axis=2)
            r, c = linear_sum_assignment(C)
            for i, j in zip(r, c):
                if C[i, j] <= self.gate:
                    tr = self.tracks[i]
                    v = (xy[j] - tr.xy) / max(t - tr.last, 1)
                    tr.v = self.alpha * v + (1 - self.alpha) * tr.v
                    tr.xy, tr.last, tr.hits = xy[j], t, tr.hits + 1
                    tr.path.append((t, *xy[j]))
                    matched_t.add(i)
                    matched_d.add(j)
        for j in range(len(xy)):
            if j not in matched_d:
                self.tracks.append(Track(self.next_id, xy[j], np.zeros(2), t, t, path=[(t, *xy[j])]))
                self.next_id += 1
        alive = []
        for tr in self.tracks:
            (alive if t - tr.last <= self.max_miss else self.finished).append(tr)
        self.tracks = alive
        return [tr for tr in self.tracks if tr.last == t and tr.hits >= self.min_hits]

    def all_tracks(self):
        return [tr for tr in self.finished + self.tracks if tr.hits >= self.min_hits]


@dataclass
class Zone:
    name: str
    x: tuple
    y: tuple

    def contains(self, xy):
        xy = np.asarray(xy).reshape(-1, 2)
        return (xy[:, 0] >= self.x[0]) & (xy[:, 0] < self.x[1]) & (xy[:, 1] >= self.y[0]) & (xy[:, 1] < self.y[1])


def default_zones() -> list[Zone]:
    x0, y0 = ORIGIN_CM
    W, H = GRID[0] * CELL_CM, GRID[1] * CELL_CM
    return [Zone(f"zone {k + 1}", (x0, x0 + W), (y0 + k * H / 3, y0 + (k + 1) * H / 3)) for k in range(3)]


def zone_counts(xy, zones) -> dict:
    return {z.name: int(z.contains(xy).sum()) for z in zones}


def line_crossings(tracks, y_line) -> dict:
    """Crossings of the line y = y_line (cm) by anonymous tracks, in each direction."""
    up = down = 0
    for tr in tracks:
        ys = [p[2] for p in tr.path]
        for a, b in zip(ys, ys[1:]):
            if a < y_line <= b:
                down += 1
            elif b < y_line <= a:
                up += 1
    return {"forward": down, "backward": up}


def dwell_seconds(tracks, zones, fps) -> dict:
    """Total and mean time anonymous tracks spend in each zone."""
    out = {}
    for z in zones:
        per = []
        for tr in tracks:
            pts = np.array([p[1:] for p in tr.path])
            ts = np.array([p[0] for p in tr.path])
            inside = z.contains(pts)
            if inside.any():
                per.append(float(np.diff(ts)[inside[:-1]].sum()) / fps if len(ts) > 1 else 0.0)
        out[z.name] = {"tracks": len(per), "total_s": float(np.sum(per)), "mean_s": float(np.mean(per)) if per else 0.0}
    return out


def heatmap(points: np.ndarray, cell=25.0) -> np.ndarray:
    """Occupancy counts on a cell x cell cm grid over the area."""
    W, H = GRID[0] * CELL_CM, GRID[1] * CELL_CM
    hm, _, _ = np.histogram2d(np.asarray(points)[:, 1] if len(points) else [], np.asarray(points)[:, 0] if len(points) else [],
                              bins=(int(H // cell), int(W // cell)),
                              range=((ORIGIN_CM[1], ORIGIN_CM[1] + H), (ORIGIN_CM[0], ORIGIN_CM[0] + W)))
    return hm


def export_yolo(root, out, train_until=TEST_FROM, workers=16) -> dict:
    """Per-view person boxes of the training frames as a YOLO dataset; test frames are never exported."""
    import shutil
    import yaml
    from concurrent.futures import ThreadPoolExecutor
    out = Path(out)
    if out.exists():
        shutil.rmtree(out)
    jobs = []
    for f in FRAMES:
        split = "train" if f < train_until - 200 else "val" if f < train_until else None
        if split is None:
            continue
        ann = load_annotations(root, f)
        for cam in range(len(CAMS)):
            boxes = [(v["xmin"], v["ymin"], v["xmax"], v["ymax"]) for p in ann for v in p["views"]
                     if v["viewNum"] == cam and v["xmin"] >= 0]
            jobs.append((split, cam, f, boxes))

    def write(job):
        split, cam, f, boxes = job
        stem = f"C{cam + 1}_{f:08d}"
        src = image_path(root, cam, f)
        dst = out / "images" / split / f"{stem}.png"
        dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            import os
            os.symlink(src, dst)
        except OSError:
            shutil.copyfile(src, dst)
        lines = []
        for x1, y1, x2, y2 in boxes:
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(1920, x2), min(1080, y2)
            if x2 - x1 > 2 and y2 - y1 > 2:
                lines.append(f"0 {(x1 + x2) / 3840:.6f} {(y1 + y2) / 2160:.6f} {(x2 - x1) / 1920:.6f} {(y2 - y1) / 1080:.6f}")
        lab = out / "labels" / split / f"{stem}.txt"
        lab.parent.mkdir(parents=True, exist_ok=True)
        lab.write_text("\n".join(lines) + ("\n" if lines else ""))

    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(write, jobs))
    (out / "data.yaml").write_text(yaml.safe_dump({"path": str(out.resolve()), "train": "images/train",
                                                   "val": "images/val", "names": {0: "person"}}, sort_keys=False))
    return {s: sum(1 for j in jobs if j[0] == s) for s in ("train", "val")}


def ground_mot(track_frames: dict, root, frames, dist=100.0) -> dict:
    """MOTA and IDF1 on the ground plane against WILDTRACK person ids, `dist` cm match radius."""
    import motmetrics as mm
    acc = mm.MOTAccumulator(auto_id=True)
    for f in frames:
        ann = load_annotations(root, f)
        g = position_to_cm([p["positionID"] for p in ann]).reshape(-1, 2)
        tr = track_frames.get(f, [])
        h = np.array([t[1:] for t in tr]).reshape(-1, 2)
        D = np.linalg.norm(g[:, None] - h[None], axis=2) if len(g) and len(h) else np.zeros((len(g), len(h)))
        D = np.where(D <= dist, D, np.nan)
        acc.update([p["personID"] for p in ann], [int(t[0]) for t in tr], D)
    s = mm.metrics.create().compute(acc, metrics=["mota", "idf1", "num_switches", "mostly_tracked", "num_unique_objects"])
    return {k: float(v) for k, v in s.iloc[0].items()}


def count_agreement(pred_frames, gt_frames, zones) -> dict:
    """Per-frame and per-zone count error against ground truth: the 'counts match manual counts' exit test."""
    tot, per_zone = [], {z.name: [] for z in zones}
    for p, g in zip(pred_frames, gt_frames):
        tot.append(abs(len(p) - len(g)))
        pc, gc = zone_counts(np.asarray(p).reshape(-1, 2), zones), zone_counts(np.asarray(g).reshape(-1, 2), zones)
        for z in zones:
            per_zone[z.name].append(abs(pc[z.name] - gc[z.name]))
    mean_gt = float(np.mean([len(g) for g in gt_frames])) if gt_frames else 0.0
    return {"count_mae": float(np.mean(tot)), "mean_people": mean_gt,
            "count_rel_err": float(np.mean(tot)) / max(mean_gt, 1e-9),
            **{f"{k}_mae": float(np.mean(v)) for k, v in per_zone.items()}}
