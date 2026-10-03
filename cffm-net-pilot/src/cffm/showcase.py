"""Scene picking, high-res pairs and inline display for the showcase notebooks."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

RAW_DIRS = {"llvip": ("visible", "infrared"), "m3fd": ("Vis", "Ir")}


def counts(root, split="val"):
    """{stem: number of labelled objects} from the converted YOLO labels."""
    lab = Path(root) / "labels" / "visible" / split
    return {f.stem: sum(1 for l in f.read_text().splitlines() if l.strip()) for f in lab.glob("*.txt")}


def crowded(root, split="val", k=6, block=200):
    """The k most crowded frames, at most one per block of consecutive file names (one per video sequence)."""
    c = counts(root, split)
    seen, out = set(), []
    for stem, n in sorted(c.items(), key=lambda kv: -kv[1]):
        key = int(stem) // block if stem.isdigit() else stem
        if key not in seen:
            seen.add(key)
            out.append(stem)
        if len(out) == k:
            break
    return out


def busiest_window(root, split="val", length=90):
    """Start stem of the run of `length` consecutive frames with the most objects."""
    c = counts(root, split)
    stems = sorted((int(s), n) for s, n in c.items() if s.isdigit())
    best, best_n = None, -1
    for i in range(len(stems) - length):
        if stems[i + length - 1][0] - stems[i][0] != length - 1:
            continue
        n = sum(x[1] for x in stems[i:i + length])
        if n > best_n:
            best, best_n = stems[i][0], n
    return [f"{best + j:06d}" for j in range(length)] if best is not None else []


def raw_root(env, name):
    from .pipeline import locate_raw
    try:
        return locate_raw(name, env)
    except Exception:
        return None


def pair_paths(env, name, stem, split="val"):
    """(visible, thermal) at full resolution if the raw data is attached, else the converted 640 px pair."""
    raw = raw_root(env, name)
    if raw is not None:
        v, i = RAW_DIRS[name]
        sub = ("test" if split == "val" else "train") if name == "llvip" else ""
        for ext in (".jpg", ".png", ".bmp"):
            vp, ip = raw / v / sub / f"{stem}{ext}", raw / i / sub / f"{stem}{ext}"
            if vp.exists() and ip.exists():
                return vp, ip
    conv = env.data / name / "images"
    return conv / "visible" / split / f"{stem}.jpg", conv / "infrared" / split / f"{stem}.jpg"


def gt_boxes(env, name, stem, shape, split="val"):
    """Ground-truth boxes (n, 6) at the image's own resolution, from the normalised YOLO labels."""
    f = env.data / name / "labels" / "visible" / split / f"{stem}.txt"
    h, w = shape[:2]
    rows = []
    for line in f.read_text().splitlines() if f.exists() else []:
        c, x, y, bw, bh = map(float, line.split())
        rows.append([(x - bw / 2) * w, (y - bh / 2) * h, (x + bw / 2) * w, (y + bh / 2) * h, 1.0, c])
    return np.array(rows, np.float32).reshape(-1, 6)


def subject_log(det, names, share=None, ids=None):
    """One row per subject: ID, class, confidence, box size and the thermal share inside its box."""
    import pandas as pd
    rows = []
    for j, (x1, y1, x2, y2, c, k) in enumerate(det[np.argsort(-det[:, 4])] if len(det) else []):
        row = {"subject": f"{(ids[j] if ids else j + 1):02d}", "class": names[int(k)].upper(), "conf": round(float(c), 3),
               "box px": f"{int(x2 - x1)}x{int(y2 - y1)}"}
        if share is not None:
            patch = share[max(0, int(y1)):int(y2) + 1, max(0, int(x1)):int(x2) + 1]
            row["thermal trust"] = round(float(patch.mean()), 2) if patch.size else None
        rows.append(row)
    return pd.DataFrame(rows).set_index("subject") if rows else pd.DataFrame()


def show(rgb, width=None, quality=90):
    """Display an RGB array inline as a compact JPEG (keeps notebooks small enough for GitHub)."""
    from IPython.display import Image, display
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, quality])
    display(Image(data=buf.tobytes(), format="jpeg", width=width or min(rgb.shape[1], 1400)))


def save(rgb, path, quality=92):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    params = [cv2.IMWRITE_JPEG_QUALITY, quality] if path.suffix.lower() in (".jpg", ".jpeg") else []
    cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), params)
    return path


def find_sequence(env, name="MOT17-04"):
    """Frame folder (img1/) of a MOT sequence, under data/raw or an attached input."""
    from .pipeline import INPUT_ROOTS
    for root in [env.raw, *INPUT_ROOTS]:
        if not Path(root).exists():
            continue
        for k in range(1, 7):
            hits = sorted(Path(root).glob("*/" * (k - 1) + f"{name}*/img1"))
            if hits:
                return hits[0]
    raise FileNotFoundError(f"{name} not found; attach MOT17 (see notebook 01)")


def track(model, frames, conf=0.3, imgsz=1280, classes=(0, 1, 2, 3, 5, 7), tracker="bytetrack.yaml", trail=40, **kw):
    """ByteTrack over a frame sequence with an Ultralytics model. Returns (boards, distinct ids)."""
    from . import hud
    boards, trails, seen = [], {}, set()
    for j, f in enumerate(frames):
        im = cv2.imread(str(f)) if not isinstance(f, np.ndarray) else f
        r = model.track(im, persist=j > 0, conf=conf, imgsz=imgsz, classes=list(classes), tracker=tracker,
                        verbose=False)[0].boxes
        keep = r.id is not None
        ids = r.id.int().tolist() if keep else []
        det = np.concatenate([r.xyxy.cpu().numpy(), r.conf.cpu().numpy()[:, None], r.cls.cpu().numpy()[:, None]], 1)
        det = det if keep else det[:0]
        for k, row in zip(ids, det):
            t = trails.setdefault(k, hud.Trail())
            t.cls = int(row[5])
            t.append(((row[0] + row[2]) / 2, row[3]))
            del t[:-trail]
        seen.update(ids)
        boards.append(hud.render(im, det, model.names, ids=ids or None, trails={k: trails[k] for k in ids}, frame=j,
                                 **kw))
    return boards, len(seen)


def degrade(pair, kind, shift_px=None):
    """Sensor faults on a full-resolution pair: visible_drop, visible_dark, thermal_drop, thermal_shift."""
    p = pair.copy()
    if kind == "visible_drop":
        p[..., :3] = 0
    elif kind == "visible_dark":
        noise = np.random.default_rng(0).normal(0, 5, p[..., :3].shape)
        p[..., :3] = np.clip(p[..., :3] * 0.15 + noise, 0, 255).astype(np.uint8)
    elif kind == "thermal_drop":
        p[..., 3] = 0
    elif kind == "thermal_shift":
        s = shift_px or round(8 * pair.shape[1] / 640)
        p[..., 3] = np.roll(p[..., 3], (s, s), axis=(0, 1))
    return p
