"""
CFFM-Net as an Ultralytics model.

The trick that keeps this file short: we do NOT rewrite YOLO26. We subclass
Ultralytics' DetectionModel, let it build the normal single-stream network from
the YAML (backbone + neck + NMS-free head), and then add three things:

  * `self.thermal`  a second copy of the backbone (layers 0..10) for the
                    thermal image, initialised from the same COCO weights
  * `self.fusers`   one fusion block per pyramid level the neck consumes
                    (P2, P3, P4, P5 = backbone layers 2, 4, 6, 10 in YOLO26)
  * a forward pass that runs both backbones side by side, swaps the fused
    features in at those four layers, and then runs the untouched YOLO26 neck
    and head on top

Because the head, the loss, the label assignment and the checkpoint format are
YOLO26's own, the Ultralytics trainer and validator work unchanged. The dual
behaviour is switched on only after the parent constructor has finished
(`self._dual`), so the parent's stride probe sees an ordinary YOLO.

Input convention: a (b, 4, H, W) tensor = visible BGR (3) + thermal (1). The
thermal channel is repeated to 3 so the COCO-pretrained stem can read it.
A (b, 3, H, W) input also works (the thermal stream then sees the grey
average); Ultralytics uses that for FLOPs counting and warm-up.
"""
from __future__ import annotations

from copy import deepcopy

import torch
import torch.nn as nn
from ultralytics.nn.tasks import DetectionModel, yaml_model_load

from .blocks import build_fuser

FUSION_KINDS = ("concat", "cffm")


class DualStreamDetectionModel(DetectionModel):
    """Two YOLO26 backbones, per-level fusion, one YOLO26 neck and head.

    cfg must be a model YAML (path or dict) with three extra keys:
        dual_stream: true
        fusion:      'concat' (baseline) or 'cffm' (CFFM-Net)
        fusion_cfg:  dict of CMFM options (gate, mixer, d_state, directions, ...)
    """

    def __init__(self, cfg="cffm-net-n.yaml", nc=None, verbose=True):
        self._dual = False                       # parent __init__ runs a stride probe: keep it single-stream
        d = cfg if isinstance(cfg, dict) else yaml_model_load(cfg)
        if not d.get("scale") and not isinstance(cfg, dict):
            # Ultralytics guesses the scale from file names like 'yolo26n.yaml'; ours says it explicitly
            import yaml
            d["scale"] = (yaml.safe_load(open(cfg, encoding="utf-8")) or {}).get("scale", "n")
        super().__init__(d, ch=3, nc=nc, verbose=False)
        self.fusion = d.get("fusion", "cffm")
        self.fusion_cfg = dict(d.get("fusion_cfg") or {})
        assert self.fusion in FUSION_KINDS, f"fusion must be one of {FUSION_KINDS}"

        # where the backbone ends, and which backbone outputs the neck reads
        self.nb = len(self.yaml["backbone"]) - 1                         # last backbone layer index (10)
        assert all(m.f == -1 for m in self.model[: self.nb + 1]), "backbone must be a plain chain"
        head_refs = {j for m in self.model[self.nb + 1:] for j in ([m.f] if isinstance(m.f, int) else m.f)}
        self.fuse_idx = sorted({j for j in head_refs if 0 <= j <= self.nb} | {self.nb})

        # probe channels and strides of the fused layers with a dummy pass
        with torch.no_grad():
            x, info = torch.zeros(1, 3, 256, 256), {}
            for m in self.model[: self.nb + 1]:
                x = m(x)
                if m.i in self.fuse_idx:
                    info[m.i] = (x.shape[1], 256 // x.shape[-1])             # (channels, stride)
        self.levels = {i: int(round(torch.log2(torch.tensor(float(s))).item())) for i, (_, s) in info.items()}

        self.thermal = deepcopy(self.model[: self.nb + 1])                 # second backbone, same architecture
        self.fusers = nn.ModuleDict({
            str(i): build_fuser(self.fusion, info[i][0], self.levels[i], self.fusion_cfg) for i in self.fuse_idx
        })
        self.sensor_flags = None                 # optional (b, 2) availability bits, see blocks.ReliabilityHead
        self._dual = True
        if verbose:
            self.info()

    # ------------------------------------------------------------------ #
    @staticmethod
    def split_modalities(x: torch.Tensor):
        """(b, 4|6|3, H, W) -> visible (b, 3, H, W), thermal (b, 3, H, W)."""
        c = x.shape[1]
        if c == 4:                                                       # visible BGR + thermal grey
            return x[:, :3], x[:, 3:4].expand(-1, 3, -1, -1)
        if c == 6:                                                       # visible + 3-channel thermal
            return x[:, :3], x[:, 3:6]
        if c == 3:                                                       # warm-up / FLOPs probe
            return x, x.mean(1, keepdim=True).expand(-1, 3, -1, -1)
        raise ValueError(f"expected 3, 4 or 6 input channels, got {c}")

    def _predict_once(self, x, profile=False, embed=None):
        if not self._dual:
            return super()._predict_once(x, profile=profile, embed=embed)
        xv, xt = self.split_modalities(x)
        y = []
        # 1) both backbones, layer by layer; fuse where the neck will read
        for mv, mt in zip(self.model[: self.nb + 1], self.thermal):
            xv, xt = mv(xv), mt(xt)
            if mv.i in self.fuse_idx:
                out = self.fusers[str(mv.i)](xv, xt, self.sensor_flags)
            else:
                out = xv
            y.append(out if mv.i in self.save else None)
        x = out                                                          # fused P5 feeds the neck
        # 2) the original YOLO26 neck and head, untouched
        for m in self.model[self.nb + 1:]:
            if m.f != -1:
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
            x = m(x)
            y.append(x if m.i in self.save else None)
        return x

    # ------------------------------------------------------------------ #
    def load(self, weights, verbose=True):
        """Load YOLO weights, then start the thermal backbone as a copy of the visible one.

        Loading a COCO checkpoint fills `model.*` only. If the checkpoint is one of
        ours (it already has `thermal.*`), the thermal weights come from it instead.
        """
        super().load(weights, verbose=verbose)
        src = (weights.get("ema") or weights["model"]) if isinstance(weights, dict) else weights
        has_thermal = any(k.startswith("thermal.") for k in src.state_dict())
        if not has_thermal:
            self.thermal.load_state_dict(self.model[: self.nb + 1].state_dict())
        return self


def describe(model: DualStreamDetectionModel) -> str:
    """One-line human summary of what was built, for notebooks and logs."""
    lv = ", ".join(f"P{model.levels[i]}<-{type(model.fusers[str(i)]).__name__}" for i in model.fuse_idx)
    n = sum(p.numel() for p in model.parameters()) / 1e6
    return f"{model.fusion} | {lv} | {n:.2f}M params"
