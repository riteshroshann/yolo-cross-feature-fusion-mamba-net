"""Training, evaluation and the test-time degradation probe.

One trainer and one validator score every run, so baselines and CFFM-Net share all the code (C5).
"""
from __future__ import annotations

import json
import math
from copy import copy
from pathlib import Path

import torch
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer, DetectionValidator
from ultralytics.nn.tasks import DetectionModel, yaml_model_load
from ultralytics.utils import LOGGER, RANK

from . import data as cdata  # also installs the paired image reader
from .model import DualStreamDetectionModel

SIZE_BINS = {  # (min, max) area in pixels^2, at the resolution the model sees
    "all": (0, 1e10),
    "vt": (0, 8 ** 2),          # AI-TOD "very tiny"
    "t": (8 ** 2, 16 ** 2),     # AI-TOD "tiny"
    "s": (16 ** 2, 32 ** 2),    # AI-TOD "small"
    "m": (32 ** 2, 96 ** 2),    # COCO medium
    "l": (96 ** 2, 1e10),       # COCO large
}


class DualValidator(DetectionValidator):
    def build_dataset(self, img_path, mode="val", batch=None):
        return cdata.build_dataset(self.args, img_path, batch, self.data, mode=mode, stride=self.stride)

    def coco_evaluate(self, stats, pred_json, anno_json, iou_types="bbox", suffix="Box"):
        stats = super().coco_evaluate(stats, pred_json, anno_json, iou_types, suffix)
        if not (self.args.save_json and self.gdict and len(self.jdict)):
            return stats
        try:  # AP in the AI-TOD size bands, from the same matched predictions
            from faster_coco_eval import COCOeval_faster

            anno = self._coco_api
            ev = COCOeval_faster(anno, anno.loadRes(pred_json), iouType="bbox", print_function=lambda *a, **k: None)
            ev.params.imgIds = anno.getImgIds()
            ev.params.areaRng = [list(v) for v in SIZE_BINS.values()]
            ev.params.areaRngLbl = list(SIZE_BINS)
            ev.evaluate()
            ev.accumulate()
            prec = ev.eval["precision"]                        # [iou, recall, class, area, maxdet]
            for a, lbl in enumerate(SIZE_BINS):
                p = prec[:, :, :, a, -1]
                stats[f"metrics/AP_{lbl}(B)"] = float(p[p > -1].mean()) if (p > -1).any() else float("nan")
                p50 = prec[0, :, :, a, -1]
                stats[f"metrics/AP50_{lbl}(B)"] = float(p50[p50 > -1].mean()) if (p50 > -1).any() else float("nan")
        except Exception as e:  # never let an extra metric kill a run
            LOGGER.warning(f"size-binned evaluation skipped: {e}")
        return stats


class DualTrainer(DetectionTrainer):
    def get_model(self, cfg=None, weights=None, verbose=True):
        d = cfg if isinstance(cfg, dict) else yaml_model_load(cfg)
        if d.get("dual_stream", False):
            model = DualStreamDetectionModel(cfg, nc=self.data["nc"], verbose=verbose and RANK == -1)
        else:
            model = DetectionModel(cfg, nc=self.data["nc"], ch=self.data["channels"], verbose=verbose and RANK == -1)
        if hasattr(self, "set_model_names_for_load"):
            model = self.set_model_names_for_load(model)
        if weights:
            model.load(weights)
        return model

    def build_dataset(self, img_path, mode="train", batch=None):
        m = self.model.module if hasattr(self.model, "module") else self.model
        gs = max(int(m.stride.max()), 32)
        return cdata.build_dataset(self.args, img_path, batch, self.data, mode=mode, rect=mode == "val", stride=gs)

    def get_validator(self):
        return DualValidator(self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks)


def train_run(name: str, model: str, data: str, *, pretrained: str = "yolo26n.pt", epochs: int = 30,
              imgsz: int = 640, batch: int = 16, device=None, project: str = "runs", seed: int = 0,
              workers: int = 4, **overrides):
    """One training run; returns the best checkpoint. Extra kwargs go straight to Ultralytics."""
    args = dict(model=str(model), data=str(data), pretrained=pretrained, epochs=epochs, imgsz=imgsz,
                batch=batch, device=device, project=str(project), name=name, exist_ok=True, seed=seed,
                deterministic=True, workers=workers, plots=True, val=True)
    args.update(overrides)
    trainer = DualTrainer(overrides=args)
    trainer.train()
    best = Path(trainer.save_dir) / "weights" / "best.pt"
    return best if best.exists() else Path(trainer.save_dir) / "weights" / "last.pt"


def _validator_args(data, imgsz, batch, device, project, name, split):
    return dict(data=str(data), imgsz=imgsz, batch=batch, device=device, save_json=True, plots=False,
                project=str(project), name=name, exist_ok=True, split=split, conf=0.001, verbose=False)


def _load(weights):
    """Load a checkpoint as a plain nn.Module (ours or stock), in eval mode."""
    return YOLO(str(weights)).model.eval()


def evaluate(weights, data, *, imgsz=640, batch=16, device=None, project="runs/eval", name=None, model=None,
             split="test"):
    """COCO-style evaluation with size bins; split='test' is the full val set, 'val' the per-epoch quarter."""
    name = name or Path(weights).parent.parent.name
    v = DualValidator(args=_validator_args(data, imgsz, batch, device, project, name, split))
    stats = v(model=model if model is not None else _load(weights))
    out = {k.replace("metrics/", ""): float(val) for k, val in stats.items() if isinstance(val, (int, float))}
    Path(project, name).mkdir(parents=True, exist_ok=True)
    (Path(project) / name / "metrics.json").write_text(json.dumps(out, indent=2))
    return out


PROBES = ("clean", "visible_dark", "visible_drop", "thermal_drop", "thermal_shift_4", "thermal_shift_8",
          "thermal_shift_16")


def _degrade(x: torch.Tensor, kind: str) -> torch.Tensor:
    """Apply one sensor degradation to a (b, 4, H, W) batch in [0, 1]."""
    x = x.clone()
    if kind == "clean":
        return x
    if kind == "visible_dark":          # night: 15% of the light plus sensor noise
        x[:, :3] = (x[:, :3] * 0.15 + 0.02 * torch.randn_like(x[:, :3])).clamp(0, 1)
    elif kind == "visible_drop":
        x[:, :3] = 0
    elif kind == "thermal_drop":
        x[:, 3:] = 0
    elif kind.startswith("thermal_shift_"):  # misregistration: thermal moved k pixels right and down
        k = int(kind.rsplit("_", 1)[1])
        x[:, 3:] = torch.roll(x[:, 3:], shifts=(k, k), dims=(2, 3))
    else:
        raise ValueError(f"unknown probe {kind!r}; choose from {PROBES}")
    return x


def probe(weights, data, kind: str, *, flagged: bool = False, imgsz=640, batch=16, device=None,
          project="runs/probe", seed=0, split="test"):
    """Evaluate a paired model under a test-time degradation; flagged=True also says which sensor dropped."""
    torch.manual_seed(seed)
    model = _load(weights)
    if not getattr(model, "_dual", False):
        raise ValueError("the probe needs a paired (dual-stream) model")
    handle = model.register_forward_pre_hook(lambda m, a: (_degrade(a[0], kind),) + tuple(a[1:]))
    if flagged and kind in ("visible_drop", "thermal_drop"):
        flags = torch.tensor([[0.0, 1.0]] if kind == "visible_drop" else [[1.0, 0.0]])
        model.sensor_flags = flags  # broadcast over the batch inside the reliability head
    try:
        name = f"{Path(weights).parent.parent.name}_{kind}{'_flagged' if flagged else ''}"
        return evaluate(weights, data, imgsz=imgsz, batch=batch, device=device, project=project, name=name,
                        model=model, split=split)
    finally:
        handle.remove()
        model.sensor_flags = None


def gflops(model, imgsz=640, include_scan=True) -> float:
    """GFLOPs at imgsz: Ultralytics' profiler count plus the selective scans, which profilers miss."""
    from ultralytics.utils.torch_utils import get_flops

    total = float(get_flops(model, imgsz))
    return total + (scan_gflops(model, imgsz) if include_scan else 0.0)


def scan_gflops(model, imgsz=640) -> float:
    """Element-wise GFLOPs of every GatedCrossScan: about 11 per token, channel and state (5 + 6 over two passes)."""
    from .blocks import GatedCrossScan

    fusers = getattr(model, "fusers", {})
    total = 0.0
    for i, f in fusers.items():
        scan = getattr(getattr(f, "mixer", None), "__class__", None)
        if scan is GatedCrossScan:
            s = 2 ** model.levels[int(i)]
            L = 2 * (imgsz // s) * (imgsz // s)
            total += 11 * f.mixer.K * f.mixer.d * f.mixer.n * L
    return total / 1e9


def latency_ms(model, imgsz=(512, 640), channels=4, device="cuda", half=True, warmup=50, iters=300,
               pause_s=0.0) -> dict:
    """Batch-1 forward latency (median and p95, ms) with CUDA events; pause_s=0.2 follows the dossier protocol."""
    import time

    m = model.to(device).eval()
    x = torch.rand(1, channels, *imgsz, device=device)
    if half and device != "cpu":
        m, x = m.half(), x.half()
    times = []
    with torch.inference_mode():
        for i in range(warmup + iters):
            if device != "cpu":
                s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                s.record()
                m(x)
                e.record()
                torch.cuda.synchronize()
                dt = s.elapsed_time(e)
            else:
                t0 = time.perf_counter()
                m(x)
                dt = (time.perf_counter() - t0) * 1e3
            if i >= warmup:
                times.append(dt)
            if pause_s:
                time.sleep(pause_s)
    times.sort()
    return {"median_ms": times[len(times) // 2], "p95_ms": times[math.ceil(0.95 * len(times)) - 1]}
