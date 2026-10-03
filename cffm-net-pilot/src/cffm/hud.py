"""Machine-style overlays: bracketed subjects with IDs, thermal and trust panels, motion trails."""
from __future__ import annotations

import time
from functools import lru_cache

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

BG = (8, 10, 13)
INK = (8, 10, 13)
WHITE = (236, 239, 242)
MUTED = (128, 138, 150)
CLASS_RGB = {"person": (255, 204, 0), "people": (255, 204, 0), "car": (0, 212, 255), "bus": (255, 124, 40),
             "truck": (255, 124, 40), "motorcycle": (255, 70, 190), "lamp": (120, 255, 128)}
FALLBACK = [(255, 204, 0), (0, 212, 255), (255, 124, 40), (255, 70, 190), (120, 255, 128), (176, 136, 255)]
VIS_RGB, THR_RGB = np.array([40, 128, 255]), np.array([255, 128, 24])


@lru_cache(None)
def font(size: int, bold: bool = False, mono: bool = True):
    from matplotlib import font_manager
    fam = "DejaVu Sans Mono" if mono else "DejaVu Sans"
    path = font_manager.findfont(font_manager.FontProperties(family=fam, weight="bold" if bold else "normal"))
    return ImageFont.truetype(path, size)


def color(name, k=0):
    return CLASS_RGB.get(str(name).lower(), FALLBACK[int(k) % len(FALLBACK)])


def _tw(d, text, f):
    l, t, r, b = d.textbbox((0, 0), text, font=f)
    return r - l, b - t


def enhance(bgr):
    """CLAHE on lightness for dark scenes; returns (image, was_enhanced)."""
    if cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).mean() > 70:
        return bgr, False
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    lab[..., 0] = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR), True


def thermal_rgb(gray):
    lo, hi = np.percentile(gray, (1, 99.5))
    g = np.clip((gray.astype(np.float32) - lo) / max(hi - lo, 1) * 255, 0, 255).astype(np.uint8)
    return cv2.cvtColor(cv2.applyColorMap(g, cv2.COLORMAP_INFERNO), cv2.COLOR_BGR2RGB)


def unletterbox(m, shape, imgsz=640):
    """Map at the letterboxed size -> map at the original (h, w)."""
    h, w = shape[:2]
    r = min(imgsz / h, imgsz / w, 1.0)
    nh, nw = round(h * r), round(w * r)
    top, left = round((imgsz - nh) / 2 - 0.1), round((imgsz - nw) / 2 - 0.1)
    return cv2.resize(m[top:top + nh, left:left + nw], (w, h), interpolation=cv2.INTER_LINEAR)


def trust_share(maps, shape, imgsz=640, levels=None):
    """Thermal share r_t / (r_v + r_t) at the original size, averaged over fusion levels."""
    levels = levels or sorted(maps)
    share = np.mean([maps[l][1] / (maps[l][0] + maps[l][1] + 1e-6) for l in levels], axis=0)
    return unletterbox(share.astype(np.float32), shape, imgsz)


def trust_rgb(bgr, share):
    """Visible in grey, tinted blue where the model trusts visible and orange where it trusts thermal."""
    g = cv2.cvtColor(enhance(bgr)[0], cv2.COLOR_BGR2GRAY).astype(np.float32)[..., None] / 255
    base = np.repeat(g * 0.55, 3, axis=2) * 255
    s = share[..., None]
    tint = np.where(s > 0.5, THR_RGB, VIS_RGB).astype(np.float32)
    a = np.clip(np.abs(s - 0.5) * 2 * 0.8 + 0.04, 0, 0.8)
    return (base * (1 - a) + tint * a).clip(0, 255).astype(np.uint8)


def _style(rgb, vignette=True, scan=True):
    out = rgb.astype(np.float32)
    h, w = out.shape[:2]
    if vignette:
        yy, xx = np.mgrid[0:h, 0:w]
        r = np.sqrt(((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2)
        out *= (1 - 0.28 * np.clip(r - 0.55, 0, 1) ** 1.5)[..., None]
    if scan:
        out[::3] *= 0.93
    return out.clip(0, 255).astype(np.uint8)


def _brackets(d, x1, y1, x2, y2, rgb, s, alpha=255):
    w, h = x2 - x1, y2 - y1
    L = int(max(5 * s, min(0.3 * min(w, h), 24 * s)))
    t = max(2, round(2.2 * s))
    c = tuple(rgb) + (alpha,)
    for r in ((x1, y1, x1 + L, y1 + t), (x1, y1, x1 + t, y1 + L), (x2 - L, y1, x2, y1 + t), (x2 - t, y1, x2, y1 + L),
              (x1, y2 - t, x1 + L, y2), (x1, y2 - L, x1 + t, y2), (x2 - L, y2 - t, x2, y2), (x2 - t, y2 - L, x2, y2)):
        d.rectangle(r, fill=c)


def _subject(d, box, rgb, s):
    x1, y1, x2, y2 = box
    d.rectangle((x1, y1, x2, y2), fill=tuple(rgb) + (16,), outline=tuple(rgb) + (110,), width=max(1, round(s)))
    _brackets(d, x1, y1, x2, y2, rgb, s)
    cx = (x1 + x2) / 2
    k = 3.2 * s
    d.polygon([(cx, y2 - k), (cx + k, y2), (cx, y2 + k), (cx - k, y2)], fill=tuple(rgb) + (230,))


def _hits(r, placed):
    return any(r[0] < p[2] and p[0] < r[2] and r[1] < p[3] and p[1] < r[3] for p in placed)


def _tag(d, box, name, conf, sid, rgb, s, top_limit, size, placed, chip=False, show_conf=True):
    x1, y1, x2, y2 = box
    W, H = size
    fid, flab = font(max(8, round((8 if chip else 9.5) * s)), bold=True), font(max(9, round(9 * s)))
    sid_txt, lab = f"{sid:02d}", f"{str(name).upper()}  {conf:.2f}" if show_conf else str(name).upper()
    iw, ih = _tw(d, sid_txt, fid)
    lw, _ = _tw(d, lab, flab) if not chip else (0, 0)
    px, py = round((3 if chip else 4) * s), round((2 if chip else 3) * s)
    th = ih + 2 * py + round(2 * s)
    tw = iw + 2 * px + (lw + 2 * px if not chip else 0)
    g = round(3 * s)
    cands = [(x1, y1 - th - g), (x1, y2 + g), (x2 + g, y1), (x1 - tw - g, y1),
             (x1, y1 - 2 * th - 4 * g), (x1, y2 + th + 3 * g), (x2 + g, y1 - th - g), (x1 - tw - g, y1 - th - g)]
    ok = [(min(max(0, x), W - tw), y) for x, y in cands if top_limit <= y <= H - th]
    free = [(x, y) for x, y in ok if not _hits((x, y, x + tw, y + th), placed)]
    if not free:
        if chip:
            return
        free = ok[:1] or [(min(max(0, x1), W - tw), max(top_limit, y1))]
    tx, ty = free[0]
    placed.append((tx, ty, tx + tw, ty + th))
    if not (x1 - g <= tx <= x2 and (ty + th <= y1 + 1 or ty >= y2 - 1)):
        ax = min(max(tx, x1), x2) if tx + tw < x1 or tx > x2 else (x1 + x2) / 2
        ay = y1 if ty + th <= y1 else y2 if ty >= y2 else (y1 + y2) / 2
        bx = tx + tw if tx + tw <= x1 else tx if tx >= x2 else min(max(ax, tx), tx + tw)
        by = ty + th / 2
        d.line([(ax, ay), (bx, by)], fill=tuple(rgb) + (200,), width=max(1, round(s)))
    d.rectangle((tx, ty, tx + iw + 2 * px, ty + th), fill=tuple(rgb) + (240,))
    if chip:
        d.text((tx + px, ty + py - round(1 * s)), sid_txt, font=fid, fill=INK + (255,))
        return
    d.text((tx + px, ty + py - round(1 * s)), sid_txt, font=fid, fill=INK + (255,))
    lx = tx + iw + 2 * px
    d.rectangle((lx, ty, lx + lw + 2 * px, ty + th), fill=INK + (205,))
    d.text((lx + px, ty + py), lab, font=flab, fill=WHITE + (255,))
    bar = round(2 * s)
    d.rectangle((lx, ty + th - bar, lx + (lw + 2 * px) * float(conf), ty + th), fill=tuple(rgb) + (255,))


def _trail(d, pts, rgb, s):
    pts = [tuple(map(float, p)) for p in pts]
    n = len(pts)
    for i in range(1, n):
        a = int(40 + 215 * (i / n) ** 1.6)
        d.line([pts[i - 1], pts[i]], fill=tuple(rgb) + (a,), width=max(2, round(2.2 * s)))
    if n >= 4:
        v = np.subtract(pts[-1], pts[-4])
        speed = np.hypot(*v)
        if speed > 1.5 * s:
            u = v / speed
            L = float(np.clip(speed * 4, 14 * s, 60 * s))
            tip = np.add(pts[-1], u * L)
            d.line([pts[-1], tuple(tip)], fill=tuple(rgb) + (255,), width=max(2, round(2.2 * s)))
            nrm, hw, hl = np.array([-u[1], u[0]]), 5 * s, 9 * s
            d.polygon([tuple(tip + u * hl * 0.3), tuple(tip - u * hl + nrm * hw), tuple(tip - u * hl - nrm * hw)],
                      fill=tuple(rgb) + (255,))


def _panel_frame(d, w, h, label, s, sub=None):
    m, L, t = round(7 * s), round(16 * s), max(1, round(1.4 * s))
    c = WHITE + (170,)
    for (x, y, dx, dy) in ((m, m, 1, 1), (w - m, m, -1, 1), (m, h - m, 1, -1), (w - m, h - m, -1, -1)):
        d.line([(x, y), (x + dx * L, y)], fill=c, width=t)
        d.line([(x, y), (x, y + dy * L)], fill=c, width=t)
    f = font(max(9, round(9.5 * s)), bold=True)
    tw, th = _tw(d, label, f)
    px, py = round(6 * s), round(4 * s)
    x0, y0 = m + round(6 * s), m + round(6 * s)
    d.rectangle((x0, y0, x0 + tw + 2 * px, y0 + th + 2 * py), fill=INK + (190,))
    d.text((x0 + px, y0 + py - round(1 * s)), label, font=f, fill=WHITE + (255,))
    if sub:
        fs = font(max(8, round(8 * s)))
        d.text((x0 + tw + 2 * px + round(6 * s), y0 + py), sub, font=fs, fill=WHITE + (200,))
    return y0 + th + 2 * py + round(4 * s)


def _overlay(rgb, det, names, ids, trails, s, label, sub=None, tags=True, gt=None, show_conf=True):
    im = Image.fromarray(rgb).convert("RGBA")
    ov = Image.new("RGBA", im.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)
    top = _panel_frame(d, im.width, im.height, label, s, sub)
    for b in (gt if gt is not None else []):
        x1, y1, x2, y2 = map(float, b[:4])
        d.rectangle((x1, y1, x2, y2), outline=WHITE + (150,), width=max(1, round(s)))
    if trails:
        for k, pts in trails.items():
            if len(pts) > 1:
                cls = pts.cls if hasattr(pts, "cls") else 0
                _trail(d, pts, color(names.get(cls, ""), cls) if isinstance(names, dict) else FALLBACK[0], s)
    order = list(np.argsort(-det[:, 4])) if len(det) else []
    for i in order:
        k = det[i][5]
        _subject(d, det[i][:4], color(names.get(int(k), ""), k), s)
    if tags:
        placed = []
        big = sorted(order, key=lambda i: -(det[i][3] - det[i][1]) * det[i][4])
        full = set(big[:8]) & {i for i in order if det[i][3] - det[i][1] >= 22 * s}
        for rank, i in enumerate(order):
            x1, y1, x2, y2, c, k = det[i][:6]
            name = names.get(int(k), "object")
            sid = int(ids[i]) if ids is not None else rank + 1
            _tag(d, (x1, y1, x2, y2), name, c, sid, color(name, k), s, top, im.size, placed, chip=i not in full,
                 show_conf=show_conf)
    return np.array(Image.alpha_composite(im, ov).convert("RGB"))


class Trail(list):
    cls = 0


def render(image, det, names, *, share=None, ids=None, trails=None, panels=None, frame=None, latency_ms=None,
           panel_w=800, gt=None, title="CFFM-NET", subtitle="MULTISPECTRAL SCENE ANALYSIS", style=True, show_conf=True):
    """One analysis board: header strip over side-by-side panels. Returns RGB."""
    det = np.asarray(det, dtype=np.float32).reshape(-1, 6) if len(det) else np.zeros((0, 6), np.float32)
    has_ir = image.ndim == 3 and image.shape[2] == 4
    panels = panels or (("visible", "thermal", "trust") if has_ir and share is not None else
                        ("visible", "thermal") if has_ir else ("visible",))
    h0, w0 = image.shape[:2]
    k = panel_w / w0
    ph = round(h0 * k)
    s = panel_w / 640
    d2 = det.copy()
    d2[:, :4] *= k
    tr2 = None
    if trails:
        tr2 = {}
        for key, pts in trails.items():
            t = Trail([(x * k, y * k) for x, y in pts])
            t.cls = getattr(pts, "cls", 0)
            tr2[key] = t
    gt2 = None if gt is None else np.asarray(gt, np.float32)[:, :4] * k
    bgr = cv2.resize(image[..., :3], (panel_w, ph), interpolation=cv2.INTER_AREA if k < 1 else cv2.INTER_CUBIC)
    names_d = names if isinstance(names, dict) else dict(enumerate(names))
    tiles = []
    for p in panels:
        if p == "visible":
            vis, enh = enhance(bgr)
            rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
            tile = _overlay(_style(rgb, scan=False) if style else rgb, d2, names_d, ids, tr2, s,
                            "VISIBLE", "display-enhanced" if enh else None, gt=gt2,
                            show_conf=show_conf)
        elif p == "thermal":
            ir = cv2.resize(image[..., 3], (panel_w, ph), interpolation=cv2.INTER_AREA if k < 1 else cv2.INTER_CUBIC)
            rgb = thermal_rgb(ir)
            tile = _overlay(_style(rgb) if style else rgb, d2, names_d, ids, tr2, s, "THERMAL", "LWIR", show_conf=show_conf)
        elif p == "trust":
            sh = cv2.resize(share, (panel_w, ph), interpolation=cv2.INTER_LINEAR)
            rgb = trust_rgb(bgr, sh)
            tile = _overlay(rgb, d2, names_d, ids, None, s, "FUSION TRUST", "blue: visible  orange: thermal",
                            tags=False)
        else:
            raise ValueError(p)
        tiles.append(tile)
    gut = round(6 * s)
    W = len(tiles) * panel_w + (len(tiles) + 1) * gut
    hh = round(30 * s)
    H = hh + ph + 2 * gut
    canvas = Image.new("RGB", (W, H), BG)
    for i, t in enumerate(tiles):
        canvas.paste(Image.fromarray(t), (gut + i * (panel_w + gut), hh + gut))
    d = ImageDraw.Draw(canvas)
    fb, fr = font(round(12 * s), bold=True), font(round(10 * s))
    d.rectangle((gut, round(9 * s), gut + round(4 * s), hh - round(5 * s)), fill=FALLBACK[0])
    d.text((gut + round(10 * s), round(8 * s)), title, font=fb, fill=WHITE)
    tw, _ = _tw(d, title, fb)
    d.text((gut + round(18 * s) + tw, round(10 * s)), f"// {subtitle}", font=fr, fill=MUTED)
    counts = {}
    for row in det:
        n = names_d.get(int(row[5]), "object")
        counts[n] = counts.get(n, 0) + 1
    parts = [f"{n.upper()} x{c}" for n, c in sorted(counts.items(), key=lambda kv: -kv[1])] or ["NO SUBJECTS"]
    if latency_ms is not None:
        parts.append(f"{latency_ms:.0f} MS")
    if frame is not None:
        parts.append(f"FRAME {frame:04d}")
    parts.append(time.strftime("%H:%M:%S"))
    right = "   ".join(parts)
    rw, _ = _tw(d, right, fr)
    d.text((W - gut - rw, round(10 * s)), right, font=fr, fill=WHITE)
    return np.array(canvas)


def analyze(model, visible, thermal=None, conf=0.3, frame=None, ids=None, trails=None, **kw):
    """Detect, time and draw one pair (CFFM-Net, concat) or one photo (an Ultralytics YOLO object).

    Returns a namespace with board (RGB), det (n, 6), ms, share (thermal trust map or None) and image.
    """
    from types import SimpleNamespace

    import torch
    from .viz import make_pair, predict_pair, reliability_maps
    yolo = hasattr(model, "predict") and hasattr(model, "model") and not isinstance(model, torch.nn.Module)
    net = model.model if yolo else model
    dual = getattr(net, "_dual", False)
    if dual:
        image = make_pair(visible, thermal)
    else:
        image = visible if isinstance(visible, np.ndarray) else cv2.imread(str(visible))
        image = image[..., :3]

    def run():
        if dual:
            return predict_pair(net, image, conf=conf)[0]
        r = model.predict(image, conf=conf, verbose=False)[0].boxes
        return np.concatenate([r.xyxy.cpu().numpy(), r.conf.cpu().numpy()[:, None], r.cls.cpu().numpy()[:, None]], 1)

    sync = torch.cuda.synchronize if torch.cuda.is_available() else (lambda: None)
    if not getattr(net, "_hud_warm", False):
        run()
        net._hud_warm = True
    sync()
    t = time.time()
    det = run()
    sync()
    ms = (time.time() - t) * 1000
    share = None
    if dual and any("rv" in getattr(f, "last", {}) for f in net.fusers.values()):
        share = trust_share(reliability_maps(net, image)[0], image.shape)
    names = model.names if yolo else net.names
    board = render(image, det, names, share=share, latency_ms=ms, frame=frame, ids=ids, trails=trails, **kw)
    return SimpleNamespace(board=board, det=det, ms=ms, share=share, image=image, names=names)


def grid(images, cols=2, gap=10, bg=BG):
    """Tile RGB images (resized to the first one's width) into a grid."""
    w = images[0].shape[1]
    ims = [cv2.resize(i, (w, round(i.shape[0] * w / i.shape[1])), interpolation=cv2.INTER_AREA) for i in images]
    rows = [ims[i:i + cols] for i in range(0, len(ims), cols)]
    H = sum(max(i.shape[0] for i in r) for r in rows) + gap * (len(rows) + 1)
    W = cols * w + gap * (cols + 1)
    out = np.full((H, W, 3), bg, np.uint8)
    y = gap
    for r in rows:
        for j, im in enumerate(r):
            x = gap + j * (w + gap)
            out[y:y + im.shape[0], x:x + w] = im
        y += max(i.shape[0] for i in r) + gap
    return out


def _iou(a, b):
    tl = np.maximum(a[:, None, :2], b[None, :, :2])
    br = np.minimum(a[:, None, 2:4], b[None, :, 2:4])
    inter = np.prod(np.clip(br - tl, 0, None), axis=2)
    area = lambda x: np.prod(x[:, 2:4] - x[:, :2], axis=1)
    return inter / (area(a)[:, None] + area(b)[None] - inter + 1e-9)


class Tracker:
    """IoU tracker with constant-velocity prediction; keeps a trail of ground points per ID."""

    def __init__(self, iou_thr=0.25, max_age=10, trail=40):
        self.iou_thr, self.max_age, self.trail = iou_thr, max_age, trail
        self.tracks, self.next_id = {}, 1

    def update(self, det):
        from scipy.optimize import linear_sum_assignment
        det = np.asarray(det, np.float32).reshape(-1, 6)
        keys = list(self.tracks)
        pred = np.array([self.tracks[k]["box"] + self.tracks[k]["vel"] for k in keys]).reshape(-1, 4)
        ids = [None] * len(det)
        if len(keys) and len(det):
            iou = _iou(det[:, :4], pred)
            r, c = linear_sum_assignment(-iou)
            for i, j in zip(r, c):
                if iou[i, j] >= self.iou_thr and self.tracks[keys[j]]["cls"] == int(det[i, 5]):
                    ids[i] = keys[j]
        for i, row in enumerate(det):
            k = ids[i]
            if k is None:
                k = ids[i] = self.next_id
                self.next_id += 1
                self.tracks[k] = {"box": row[:4].copy(), "vel": np.zeros(4, np.float32), "age": 0, "hits": 0,
                                  "cls": int(row[5]), "trail": Trail()}
                self.tracks[k]["trail"].cls = int(row[5])
            t = self.tracks[k]
            t["vel"] = 0.6 * t["vel"] + 0.4 * (row[:4] - t["box"]) if t["hits"] else t["vel"]
            t["box"], t["age"], t["hits"] = row[:4].copy(), 0, t["hits"] + 1
            t["trail"].append(((row[0] + row[2]) / 2, row[3]))
            del t["trail"][:-self.trail]
        for k in keys:
            if k not in ids:
                self.tracks[k]["age"] += 1
                if self.tracks[k]["age"] > self.max_age:
                    del self.tracks[k]
        return ids

    def trails(self, ids):
        return {k: self.tracks[k]["trail"] for k in ids if k in self.tracks}


def save_video(frames, path, fps=12, width=None):
    """MP4 (H.264) or GIF depending on the suffix."""
    import imageio.v2 as iio
    if width:
        frames = [cv2.resize(f, (width, round(f.shape[0] * width / f.shape[1])), interpolation=cv2.INTER_AREA)
                  for f in frames]
    path = str(path)
    if path.endswith(".gif"):
        iio.mimsave(path, frames, duration=1 / fps, loop=0)
    else:
        frames = [f[: f.shape[0] // 2 * 2, : f.shape[1] // 2 * 2] for f in frames]
        iio.mimsave(path, frames, fps=fps, codec="libx264", quality=8, macro_block_size=1)
    return path
