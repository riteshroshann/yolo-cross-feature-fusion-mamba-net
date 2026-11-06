"""CFFM-Net as an Ultralytics model: stock YOLO26 plus a thermal backbone and per-level fusers.

Input is (b, 4, H, W), visible BGR + thermal grey; the stock trainer, loss and validator work unchanged.
"""
from __future__ import annotations

from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.tasks import DetectionModel, yaml_model_load

from . import faults
from .blocks import build_fuser

FUSION_KINDS = ("concat", "cffm")
FAULT_TRUST = (0.7, 0.4, 0.15)


class DualStreamDetectionModel(DetectionModel):
    """Two YOLO26 backbones, per-level fusion, one YOLO26 neck and head (YAML keys: fusion, fusion_cfg)."""

    def __init__(self, cfg="cffm-net-n.yaml", nc=None, verbose=True):
        self._dual = False
        d = cfg if isinstance(cfg, dict) else yaml_model_load(cfg)
        if not d.get("scale") and not isinstance(cfg, dict):
            import yaml
            d["scale"] = (yaml.safe_load(open(cfg, encoding="utf-8")) or {}).get("scale", "n")
        super().__init__(d, ch=3, nc=nc, verbose=False)
        self.fusion = d.get("fusion", "cffm")
        self.fusion_cfg = dict(d.get("fusion_cfg") or {})
        self.modality_dropout = float(d.get("modality_dropout", 0.0))
        self.fault_train = float(d.get("fault_train", 0.0))
        self.rel_loss = float(d.get("rel_loss", 0.0))
        self._rel_target = None
        self.last_rel_loss = 0.0
        assert self.fusion in FUSION_KINDS, f"fusion must be one of {FUSION_KINDS}"

        self.nb = len(self.yaml["backbone"]) - 1
        assert all(m.f == -1 for m in self.model[: self.nb + 1]), "backbone must be a plain chain"
        head_refs = {j for m in self.model[self.nb + 1:] for j in ([m.f] if isinstance(m.f, int) else m.f)}
        self.fuse_idx = sorted({j for j in head_refs if 0 <= j <= self.nb} | {self.nb})

        with torch.no_grad():
            x, info = torch.zeros(1, 3, 256, 256), {}
            for m in self.model[: self.nb + 1]:
                x = m(x)
                if m.i in self.fuse_idx:
                    info[m.i] = (x.shape[1], 256 // x.shape[-1])
        self.levels = {i: int(round(torch.log2(torch.tensor(float(s))).item())) for i, (_, s) in info.items()}

        self.thermal = deepcopy(self.model[: self.nb + 1])
        self.fusers = nn.ModuleDict({
            str(i): build_fuser(self.fusion, info[i][0], self.levels[i], self.fusion_cfg) for i in self.fuse_idx
        })
        self.sensor_flags = None
        self._dual = True
        if verbose:
            self.info()

    @staticmethod
    def split_modalities(x: torch.Tensor):
        """(b, 4|6|3, H, W) -> visible (b, 3, H, W), thermal (b, 3, H, W)."""
        c = x.shape[1]
        if c == 4:
            return x[:, :3], x[:, 3:4].expand(-1, 3, -1, -1)
        if c == 6:
            return x[:, :3], x[:, 3:6]
        if c == 3:
            return x, x.mean(1, keepdim=True).expand(-1, 3, -1, -1)
        raise ValueError(f"expected 3, 4 or 6 input channels, got {c}")

    @staticmethod
    def drop_modalities(x, p):
        """Black out visible or thermal per sample with probability p each; flag the loss for half of them."""
        b = x.shape[0]
        u = torch.rand(b, device=x.device)
        keep = torch.stack([u >= p, (u < p) | (u >= 2 * p)], 1).to(x.dtype)
        x = x.clone()
        x[:, :3] = x[:, :3] * keep[:, :1, None, None]
        x[:, 3:] = x[:, 3:] * keep[:, 1:, None, None]
        told = (torch.rand(b, 1, device=x.device) < 0.5).to(x.dtype)
        return x, keep * told + (1 - told)

    @staticmethod
    def corrupt(x, p):
        """Corrupt one camera per sample with probability p by a random benchmark fault; return trust targets."""
        b = x.shape[0]
        target = torch.ones(b, 2, device=x.device)
        x = x.clone()
        for i in torch.nonzero(torch.rand(b) < p).flatten().tolist():
            sensor = "visible" if torch.rand(1).item() < 0.5 else "thermal"
            kinds = [k for k in (faults.VISIBLE if sensor == "visible" else faults.THERMAL) if k != "drop"]
            kind = kinds[torch.randint(len(kinds), (1,)).item()]
            sev = int(torch.randint(1, 4, (1,)).item())
            x[i:i + 1] = faults.apply(x[i:i + 1], f"{sensor}_{kind}", sev, seed=int(torch.randint(1 << 30, (1,))))
            target[i, 0 if sensor == "visible" else 1] = FAULT_TRUST[sev - 1]
        return x, target

    def _predict_once(self, x, profile=False, embed=None):
        if not self._dual:
            return super()._predict_once(x, profile=profile, embed=embed)
        flags = self.sensor_flags
        self._rel_target = None
        if self.training and x.shape[1] in (4, 6):
            target = torch.ones(x.shape[0], 2, device=x.device)
            if self.fault_train > 0:
                x, target = self.corrupt(x, self.fault_train)
            if self.modality_dropout > 0:
                x, flags = self.drop_modalities(x, self.modality_dropout)
                target = target * (x.flatten(2).abs().amax(2)[:, [0, 3]] > 0).float()
            self._rel_target = target
        xv, xt = self.split_modalities(x)
        y = []
        for mv, mt in zip(self.model[: self.nb + 1], self.thermal):
            xv, xt = mv(xv), mt(xt)
            if mv.i in self.fuse_idx:
                out = self.fusers[str(mv.i)](xv, xt, flags)
                if getattr(self, "memory", None) is not None and self.memory_on:
                    out = self._remember(str(mv.i), out)
            else:
                out = xv
            y.append(out if mv.i in self.save else None)
        x = out
        for m in self.model[self.nb + 1:]:
            if m.f != -1:
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
            x = m(x)
            y.append(x if m.i in self.save else None)
        return x

    def attach_memory(self, ratio: float = 0.5, d_state: int = 8):
        """Add one Temporal State Memory per fusion level; zero-initialised, so detections start unchanged."""
        from .tsm import TemporalStateMemory
        shapes, hooks = {}, []
        for k, f in self.fusers.items():
            hooks.append(f.register_forward_hook(lambda m, a, o, k=k: shapes.__setitem__(k, o.shape[1])))
        was = self.training
        self.eval()
        with torch.no_grad():
            self._predict_once(torch.zeros(1, 4, 128, 128, device=next(self.parameters()).device))
        self.train(was)
        for h in hooks:
            h.remove()
        dev = next(self.parameters()).device
        self.memory = nn.ModuleDict({k: TemporalStateMemory(c, ratio, d_state) for k, c in shapes.items()}).to(dev)
        self.memory_on = True
        self.reset_memory()
        return self.memory

    def reset_memory(self):
        self._mem_state, self._mem_ctx = {}, {}

    def memory_context(self, tau_ratio=None, H=None, reset=None):
        """Per-frame context for the memory: time-gap ratio, homography from the previous frame, scene cut."""
        self._mem_ctx = {"tau_ratio": tau_ratio, "H": H, "reset": reset}

    def _remember(self, k, z):
        last = getattr(self.fusers[k], "last", {})
        rho = last.get("rho")
        if rho is None and "rv" in last:
            rho = torch.maximum(last["rv"], last["rt"])
        state = self._mem_state.get(k)
        if state is not None and (state.shape[0] != z.shape[0] or state.shape[1:3] != z.shape[2:]):
            state = None
        z, self._mem_state[k] = self.memory[k](z, state, rho=rho, **self._mem_ctx)
        return z

    def reliability_loss(self):
        """BCE between each gated fuser's mean reliability and the known per-camera trust target."""
        t = self._rel_target
        live = [f.live for f in self.fusers.values() if getattr(f, "live", None) is not None]
        if t is None or not live:
            return None
        loss = 0.0
        with torch.autocast(device_type=t.device.type, enabled=False):
            for rv, rt in live:
                r = torch.cat([rv.float().mean((2, 3)), rt.float().mean((2, 3))], 1).clamp(1e-4, 1 - 1e-4)
                loss = loss + F.binary_cross_entropy(r, t.float())
        return loss / len(live)

    def loss(self, batch, preds=None):
        total, items = super().loss(batch, preds)
        if self._dual and self.rel_loss > 0 and self.training:
            aux = self.reliability_loss()
            if aux is not None:
                self.last_rel_loss = float(aux.detach())
                total = total + self.rel_loss * aux * batch["img"].shape[0]
        return total, items

    def load(self, weights, verbose=True):
        """Load YOLO weights; without `thermal.*` keys the thermal backbone copies the visible one."""
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
