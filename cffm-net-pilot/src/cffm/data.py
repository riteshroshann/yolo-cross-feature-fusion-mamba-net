"""Paired visible-thermal data for Ultralytics, plus converters for LLVIP and M3FD.

A pair loads as one 4-channel image (BGR + thermal grey), so geometric augmentations never split it.
"""
from __future__ import annotations

import contextvars
import json
import os
import random
import shutil
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import yaml

import ultralytics.data.base as _ul_base
import ultralytics.data.build as _ul_build
from ultralytics.data.augment import RandomHSV
from ultralytics.data.dataset import YOLODataset

_PAIRED = contextvars.ContextVar("cffm_paired_read", default=False)
_ORIG_IMREAD = getattr(_ul_base.imread, "_cffm_original", _ul_base.imread)
SEP_VIS = (f"{os.sep}images{os.sep}visible{os.sep}", "/images/visible/")
SEP_IR = (f"{os.sep}images{os.sep}infrared{os.sep}", "/images/infrared/")


def thermal_path(visible_path: str) -> str:
    """images/visible/... -> images/infrared/... (same file name)."""
    p = str(visible_path)
    for vis, ir in zip(SEP_VIS, SEP_IR):
        if vis in p:
            head, _, tail = p.rpartition(vis)
            return head + ir + tail
    raise ValueError(f"not a paired visible path: {p}")


def read_pair(visible_path: str) -> np.ndarray | None:
    """Return a (H, W, 4) uint8 array: visible BGR + thermal grey."""
    vis = _ORIG_IMREAD(str(visible_path), flags=cv2.IMREAD_COLOR)
    ir = _ORIG_IMREAD(thermal_path(visible_path), flags=cv2.IMREAD_GRAYSCALE)
    if vis is None or ir is None:
        return None
    if ir.ndim == 3:
        ir = ir[..., 0]
    if ir.shape[:2] != vis.shape[:2]:  # converters write equal sizes; be forgiving anyway
        ir = cv2.resize(ir, (vis.shape[1], vis.shape[0]), interpolation=cv2.INTER_LINEAR)
    return np.concatenate([vis, ir[..., None]], axis=2)


def _imread(filename, flags=cv2.IMREAD_COLOR):
    return read_pair(filename) if _PAIRED.get() else _ORIG_IMREAD(filename, flags=flags)


_imread._cffm_original = _ORIG_IMREAD
_ul_base.imread = _imread  # patched on import, so every DataLoader worker gets it too


class VisibleHSV:
    """Ultralytics' HSV jitter on the visible channels, since stock RandomHSV skips 4-channel images."""

    def __init__(self, hgain: float, sgain: float, vgain: float):
        self.hsv = RandomHSV(hgain, sgain, vgain)

    def __call__(self, labels):
        img = labels["img"]
        if img.ndim == 3 and img.shape[-1] == 4:
            vis = np.ascontiguousarray(img[..., :3])
            self.hsv.apply_image({"img": vis})
            img[..., :3] = vis
        return labels


class PairedYOLODataset(YOLODataset):
    """YOLODataset whose images are visible + thermal, 4 channels."""

    def load_image(self, i, *args, **kwargs):
        token = _PAIRED.set(True)
        try:
            return super().load_image(i, *args, **kwargs)
        finally:
            _PAIRED.reset(token)

    def build_transforms(self, hyp=None):
        t = super().build_transforms(hyp)
        if self.augment and hyp is not None and (hyp.hsv_h or hyp.hsv_s or hyp.hsv_v):
            t.insert(len(t.transforms) - 1, VisibleHSV(hyp.hsv_h, hyp.hsv_s, hyp.hsv_v))  # before Format
        return t


def build_dataset(cfg, img_path, batch, data, mode="train", rect=False, stride=32):
    """Ultralytics' build_yolo_dataset, with PairedYOLODataset when the data YAML says `paired: true`."""
    if not data.get("paired", False):
        return _ul_build.build_yolo_dataset(cfg, img_path, batch, data, mode=mode, rect=rect, stride=stride)
    if getattr(cfg, "cache", None) == "disk":
        raise ValueError("cache='disk' stores single images; use cache=False or cache='ram' for paired data")
    original = _ul_build.YOLODataset
    _ul_build.YOLODataset = PairedYOLODataset  # build_yolo_dataset looks the class up by this module name
    try:
        return _ul_build.build_yolo_dataset(cfg, img_path, batch, data, mode=mode, rect=rect, stride=stride)
    finally:
        _ul_build.YOLODataset = original


def _find_dir(root: Path, names) -> Path | None:
    """First directory under root (any depth) whose name is in `names` (case-insensitive)."""
    names = {n.lower() for n in names}
    level = [Path(root)]
    while level:  # breadth-first over folders only; files are never listed twice
        nxt = []
        for d in level:
            subs = sorted(p for p in d.iterdir() if p.is_dir())
            for p in subs:
                if p.name.lower() in names:
                    return p
            nxt += subs
        level = nxt
    return None


def _parse_voc(xml_path: Path, class_names: list[str]):
    """VOC XML -> list of (class_id, x1, y1, x2, y2) in pixels. Unknown classes are skipped."""
    lookup = {n.lower(): i for i, n in enumerate(class_names)}
    out = []
    for obj in ET.parse(xml_path).getroot().iter("object"):
        name = (obj.findtext("name") or "").strip().lower()
        if name not in lookup:
            continue
        bb = obj.find("bndbox")
        x1, y1, x2, y2 = (float(bb.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax"))
        out.append((lookup[name], x1, y1, x2, y2))
    return out


def _write_pair(job):
    """Resize one pair so the long side is `imgsz`, write both images and the label."""
    vis_src, ir_src, xml, out, split, stem, names, imgsz = job
    vis = cv2.imread(str(vis_src), cv2.IMREAD_COLOR)
    ir = cv2.imread(str(ir_src), cv2.IMREAD_GRAYSCALE)
    if vis is None or ir is None:
        return None
    h, w = vis.shape[:2]
    if ir.shape[:2] != (h, w):
        ir = cv2.resize(ir, (w, h), interpolation=cv2.INTER_LINEAR)
    r = imgsz / max(h, w)
    if r < 1:
        size = (round(w * r), round(h * r))
        vis = cv2.resize(vis, size, interpolation=cv2.INTER_AREA)
        ir = cv2.resize(ir, size, interpolation=cv2.INTER_AREA)
    lines, sizes = [], []
    for c, x1, y1, x2, y2 in _parse_voc(xml, names):
        x1, x2 = max(0.0, min(x1, x2)), min(float(w), max(x1, x2))
        y1, y2 = max(0.0, min(y1, y2)), min(float(h), max(y1, y2))
        if x2 - x1 < 1 or y2 - y1 < 1:
            continue
        lines.append(f"{c} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}")
        sizes.append(((x2 - x1) * (y2 - y1)) ** 0.5 * min(r, 1.0))  # object size in pixels after resizing
    for mod, im in (("visible", vis), ("infrared", ir)):
        dst = out / "images" / mod / split / f"{stem}.jpg"
        dst.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(dst), im, [cv2.IMWRITE_JPEG_QUALITY, 95])
        lab = out / "labels" / mod / split / f"{stem}.txt"
        lab.parent.mkdir(parents=True, exist_ok=True)
        lab.write_text("\n".join(lines) + ("\n" if lines else ""))
    return sizes


def _convert(pairs, out: Path, names, imgsz, workers, meta):
    """Write all pairs, then the data YAMLs and a dataset card."""
    if out.exists():
        shutil.rmtree(out)
    jobs = [(v, i, x, out, s, stem, names, imgsz) for (v, i, x, s, stem) in pairs]
    with ThreadPoolExecutor(workers) as ex:  # OpenCV releases the GIL, threads are enough
        results = list(ex.map(_write_pair, jobs))
    sizes = [s for r in results if r for s in r]
    n_ok = {s: sum(1 for (j, r) in zip(jobs, results) if r is not None and j[4] == s) for s in ("train", "val")}
    bins = {"<8px": 0, "8-16px": 0, "16-32px": 0, "32-96px": 0, ">96px": 0}
    for s in sizes:
        k = "<8px" if s < 8 else "8-16px" if s < 16 else "16-32px" if s < 32 else "32-96px" if s < 96 else ">96px"
        bins[k] += 1
    card = {**meta, "images": n_ok, "objects": len(sizes), "object_size_bins": bins, "imgsz": imgsz,
            "names": names}
    (out / "dataset_card.json").write_text(json.dumps(card, indent=2))
    write_yamls(out, names)
    return card


def write_yamls(out: Path, names, mini_every: int = 4):
    """Paired, visible and thermal YAMLs; val is every `mini_every`-th val image, test the full val set."""
    base = {"path": str(out.resolve()), "names": dict(enumerate(names))}
    for mod in ("visible", "infrared"):
        files = sorted((out / "images" / mod / "val").glob("*.jpg"))[::mini_every]
        (out / f"val_mini_{mod}.txt").write_text("".join(f"./images/{mod}/val/{f.name}\n" for f in files))
    views = {
        "data_paired.yaml": {**base, "train": "images/visible/train", "val": "val_mini_visible.txt",
                             "test": "images/visible/val", "channels": 4, "paired": True},
        "data_visible.yaml": {**base, "train": "images/visible/train", "val": "val_mini_visible.txt",
                              "test": "images/visible/val"},
        "data_infrared.yaml": {**base, "train": "images/infrared/train", "val": "val_mini_infrared.txt",
                               "test": "images/infrared/val"},
    }
    for fname, d in views.items():
        (out / fname).write_text(yaml.safe_dump(d, sort_keys=False))


def convert_llvip(raw: str | Path, out: str | Path, imgsz=640, train_stride=3, val_stride=1, workers=8):
    """LLVIP -> paired YOLO layout; test becomes val, and train keeps every `train_stride`-th video frame."""
    raw, out = Path(raw), Path(out)
    vis_root, ir_root = _find_dir(raw, ["visible"]), _find_dir(raw, ["infrared"])
    ann = _find_dir(raw, ["Annotations", "annotation"])
    if not (vis_root and ir_root and ann):
        raise FileNotFoundError(f"LLVIP folders visible/, infrared/, Annotations/ not found under {raw}")
    pairs, xml = [], {p.stem for p in ann.iterdir()}
    for src_split, dst_split, stride in (("train", "train", train_stride), ("test", "val", val_stride)):
        files = sorted((vis_root / src_split).glob("*.jpg"))[::stride]
        ir = {p.name for p in (ir_root / src_split).iterdir()}
        for v in files:
            if v.name in ir and v.stem in xml:
                pairs.append((v, ir_root / src_split / v.name, ann / f"{v.stem}.xml", dst_split, v.stem))
    meta = {"dataset": "LLVIP", "source": "https://bupt-ai-cz.github.io/LLVIP/",
            "split": f"official train (every {train_stride}th frame) / official test (every {val_stride}th)"}
    return _convert(pairs, out, ["person"], imgsz, workers, meta)


M3FD_NAMES = ["People", "Car", "Bus", "Motorcycle", "Lamp", "Truck"]


def convert_m3fd(raw: str | Path, out: str | Path, imgsz=640, val_frac=0.2, seed=0, workers=8):
    """M3FD -> paired YOLO layout, with a seeded 80/20 split since there is no official one."""
    raw, out = Path(raw), Path(out)
    vis_root = _find_dir(raw, ["Vis", "visible"])
    ir_root = _find_dir(raw, ["Ir", "infrared"])
    ann = _find_dir(raw, ["Annotation", "Annotations", "Labels"])
    if not (vis_root and ir_root and ann):
        raise FileNotFoundError(f"M3FD folders Vis/, Ir/, Annotation/ not found under {raw}")
    exts = {".png", ".jpg", ".bmp"}
    vis = {p.stem: p for p in vis_root.iterdir() if p.suffix.lower() in exts}   # one listing per folder
    ir = {p.stem: p for p in ir_root.iterdir() if p.suffix.lower() in exts}
    xml = {p.stem for p in ann.iterdir() if p.suffix.lower() == ".xml"}
    stems = sorted(vis)
    rng = random.Random(seed)
    val = set(rng.sample(stems, round(len(stems) * val_frac)))
    pairs = [(vis[s], ir[s], ann / f"{s}.xml", "val" if s in val else "train", s)
             for s in stems if s in ir and s in xml]
    meta = {"dataset": "M3FD", "source": "https://github.com/JinyuanLiu-CV/TarDAL",
            "split": f"seeded random {int((1 - val_frac) * 100)}/{int(val_frac * 100)} (seed={seed})"}
    return _convert(pairs, out, M3FD_NAMES, imgsz, workers, meta)


def make_synthetic(out: str | Path, n_train=8, n_val=4, imgsz=320, seed=0):
    """A tiny fake paired dataset (bright squares, hotter in thermal) for smoke tests."""
    out, rng = Path(out), np.random.default_rng(seed)
    if out.exists():
        shutil.rmtree(out)
    for split, n in (("train", n_train), ("val", n_val)):
        for k in range(n):
            vis = rng.integers(0, 60, (imgsz, imgsz, 3), dtype=np.uint8)
            ir = rng.integers(0, 40, (imgsz, imgsz), dtype=np.uint8)
            lines = []
            for _ in range(rng.integers(1, 4)):
                s = int(rng.integers(12, 48))
                x, y = int(rng.integers(0, imgsz - s)), int(rng.integers(0, imgsz - s))
                vis[y:y + s, x:x + s] = 200
                ir[y:y + s, x:x + s] = 250
                lines.append(f"0 {(x + s / 2) / imgsz:.6f} {(y + s / 2) / imgsz:.6f} {s / imgsz:.6f} {s / imgsz:.6f}")
            for mod, im in (("visible", vis), ("infrared", ir)):
                p = out / "images" / mod / split / f"{k:04d}.jpg"
                p.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(p), im)
                lp = out / "labels" / mod / split / f"{k:04d}.txt"
                lp.parent.mkdir(parents=True, exist_ok=True)
                lp.write_text("\n".join(lines) + "\n")
    write_yamls(out, ["object"])
    return out
