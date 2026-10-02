"""CFFM-Net fusion blocks: CMFM, its parts, and the baseline fusers.

Shapes: b batch, c level channels, d block channels, h w spatial, L sequence length.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .scan import selective_scan

EPS = 1e-6


class LayerNorm2d(nn.Module):
    """LayerNorm over channels for (b, c, h, w) tensors."""

    def __init__(self, c: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(c))
        self.bias = nn.Parameter(torch.zeros(c))
        self.eps = eps

    def forward(self, x):
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        x = (x - mu) / torch.sqrt(var + self.eps)
        return x * self.weight[:, None, None] + self.bias[:, None, None]


def conv_bn_act(c_in: int, c_out: int, k: int = 1, groups: int = 1, act: bool = True) -> nn.Sequential:
    layers = [nn.Conv2d(c_in, c_out, k, padding=k // 2, groups=groups, bias=False), nn.BatchNorm2d(c_out)]
    if act:
        layers.append(nn.SiLU())
    return nn.Sequential(*layers)


class ReliabilityHead(nn.Module):
    """Per-pixel reliability r in (0, 1) for each modality, from [F_v, F_t, |F_v - F_t|]."""

    def __init__(self, c: int, init_bias: float = 2.0):
        super().__init__()
        self.body = conv_bn_act(3 * c, 3 * c, k=3, groups=3 * c)
        self.out = nn.Conv2d(3 * c, 2, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.constant_(self.out.bias, init_bias)

    def forward(self, fv, ft, flags=None):
        x = torch.cat([fv, ft, (fv - ft).abs()], dim=1)
        r = torch.sigmoid(self.out(self.body(x)))                    # starts at sigmoid(2) = 0.88: trust both
        if flags is not None:
            r = r * flags.to(device=r.device, dtype=r.dtype)[:, :, None, None]
        return r[:, :1], r[:, 1:]


class OffsetAlign(nn.Module):
    """Warp thermal onto visible by a learned offset of at most max_disp cells, starting at identity."""

    def __init__(self, c: int, max_disp: float = 4.0):
        super().__init__()
        self.body = conv_bn_act(2 * c, c, k=3)
        self.out = nn.Conv2d(c, 2, 3, padding=1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        self.max_disp = max_disp

    def forward(self, fv, ft):
        b, _, h, w = ft.shape
        off = torch.tanh(self.out(self.body(torch.cat([fv, ft], 1)))) * self.max_disp  # (b, 2, h, w), cells
        ys = torch.linspace(-1, 1, h, device=ft.device, dtype=ft.dtype)
        xs = torch.linspace(-1, 1, w, device=ft.device, dtype=ft.dtype)
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        grid = torch.stack([gx, gy], dim=-1).unsqueeze(0)
        scale = torch.tensor([2.0 / max(w - 1, 1), 2.0 / max(h - 1, 1)], device=ft.device, dtype=ft.dtype)
        grid = grid + off.permute(0, 2, 3, 1) * scale                # cells -> normalised coords
        return F.grid_sample(ft, grid, mode="bilinear", padding_mode="border", align_corners=True)


class GatedCrossScan(nn.Module):
    """Selective scan over interleaved v/t tokens in K directions, step size scaled by token reliability."""

    def __init__(self, d: int, d_state: int = 8, directions: int = 4, dt_rank: int | None = None,
                 dt_min: float = 1e-3, dt_max: float = 1e-1, impl: str = "auto",
                 chunk: int | None = None, use_checkpoint: bool = True):
        super().__init__()
        assert directions in (2, 4), "use 4 (rows and columns, both ways) or 2 (rows and columns, forward)"
        self.d, self.n, self.K = d, d_state, directions
        self.r = dt_rank or math.ceil(d / 16)
        self.impl, self.chunk, self.use_checkpoint = impl, chunk, use_checkpoint
        K, r, n = self.K, self.r, self.n

        bound = 1.0 / math.sqrt(d)
        self.x_proj = nn.Parameter(torch.empty(K, r + 2 * n, d).uniform_(-bound, bound))
        # softplus(dt_b) starts in [dt_min, dt_max]
        self.dt_w = nn.Parameter(torch.empty(K, d, r).uniform_(-r ** -0.5, r ** -0.5))
        dt = torch.exp(torch.rand(K, d) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min))
        self.dt_b = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))  # inverse softplus
        self.A_log = nn.Parameter(torch.log(torch.arange(1, n + 1, dtype=torch.float32)).repeat(K * d, 1))
        self.D = nn.Parameter(torch.ones(K * d))

    def _orders(self, X):
        """(b, c, h, w, 2) -> (b, K, c, L), L = 2hw, v and t interleaved at each location."""
        b, c = X.shape[:2]
        rows = X.reshape(b, c, -1)
        cols = X.permute(0, 1, 3, 2, 4).reshape(b, c, -1)
        seqs = [rows, cols] if self.K == 2 else [rows, rows.flip(-1), cols, cols.flip(-1)]
        return torch.stack(seqs, dim=1)

    def _merge(self, y, h, w):
        """(b, K, d, L) -> (b, d, h, w, 2): undo each order and sum the directions."""
        b, _, d, _ = y.shape
        as_rows = lambda s: s.reshape(b, d, h, w, 2)
        as_cols = lambda s: s.reshape(b, d, w, h, 2).permute(0, 1, 3, 2, 4)
        if self.K == 2:
            return as_rows(y[:, 0]) + as_cols(y[:, 1])
        return (as_rows(y[:, 0]) + as_rows(y[:, 1].flip(-1))
                + as_cols(y[:, 2]) + as_cols(y[:, 3].flip(-1)))

    def forward(self, xv, xt, rv, rt):
        # the whole SSM path stays in float32: over 10k tokens the state overflows float16
        with torch.autocast(device_type=xv.device.type, enabled=False):
            return self._forward(xv.float(), xt.float(), rv.float(), rt.float())

    def _forward(self, xv, xt, rv, rt):
        b, d, h, w = xv.shape
        K, n, r = self.K, self.n, self.r
        seqs = self._orders(torch.stack([xv, xt], dim=-1))           # (b, K, d, L)
        rel = self._orders(torch.stack([rv, rt], dim=-1))            # (b, K, 1, L)
        L = seqs.shape[-1]

        x_dbl = torch.einsum("bkdl,kcd->bkcl", seqs, self.x_proj.float())   # (b, K, r+2n, L)
        dts, Bs, Cs = torch.split(x_dbl, [r, n, n], dim=2)
        dts = torch.einsum("bkrl,kdr->bkdl", dts, self.dt_w.float()) + self.dt_b.float()[None, :, :, None]
        delta = F.softplus(dts) * rel                                # r -> 0 skips the token, state untouched

        A = -torch.exp(self.A_log.float())
        y = selective_scan(seqs.reshape(b, K * d, L), delta.reshape(b, K * d, L), A,
                           Bs.contiguous(), Cs.contiguous(), self.D.float(), impl=self.impl,
                           **({} if self.impl == "cuda" else
                              {"chunk": self.chunk, "use_checkpoint": self.use_checkpoint}))
        Y = self._merge(y.view(b, K, d, L), h, w)
        return Y[..., 0], Y[..., 1]


class GatedConvMixer(nn.Module):
    """The C4 control: same interface as GatedCrossScan, gated depthwise conv instead of the SSM."""

    def __init__(self, d: int, kernel: int = 7, expand: float = 2.0):
        super().__init__()
        hid = int(2 * d * expand)
        self.inp = nn.Conv2d(2 * d, hid, 1)
        self.dw = nn.Conv2d(hid, hid, kernel, padding=kernel // 2, groups=hid)
        self.gate = nn.Conv2d(2 * d, hid, 1)
        self.out = nn.Conv2d(hid, 2 * d, 1)

    def forward(self, xv, xt, rv, rt):
        x = torch.cat([xv * rv, xt * rt], dim=1)                     # no step size here, so gate the input
        y = self.out(self.dw(self.inp(x)) * torch.sigmoid(self.gate(x)))
        return y.chunk(2, dim=1)


class CMFM(nn.Module):
    """Cross-Modal Fusion Mamba block: (F_v, F_t) -> Z, all (b, c, h, w)."""

    def __init__(self, c: int, ratio: float = 0.5, d_state: int = 8, directions: int = 4,
                 mixer: str = "ssm", gate: bool = True, align: bool = True, impl: str = "auto",
                 use_checkpoint: bool = True, gconv_expand: float = 2.0):
        super().__init__()
        d = max(16, int(round(c * ratio)))
        self.gate, self.c, self.d = gate, c, d
        self.rel = ReliabilityHead(c) if gate else None   # no unused parameters when gating is off
        self.align = OffsetAlign(c) if align else None
        # the 3x3 depthwise conv plays the role of Mamba's short causal conv
        self.phi_v = nn.Sequential(conv_bn_act(c, d, 1), conv_bn_act(d, d, 3, groups=d))
        self.phi_t = nn.Sequential(conv_bn_act(c, d, 1), conv_bn_act(d, d, 3, groups=d))
        if mixer == "ssm":
            self.mixer = GatedCrossScan(d, d_state=d_state, directions=directions, impl=impl,
                                        use_checkpoint=use_checkpoint)
        elif mixer == "gconv":
            self.mixer = GatedConvMixer(d, expand=gconv_expand)
        else:
            raise ValueError(f"unknown mixer {mixer!r}")
        self.norm = LayerNorm2d(2 * d)
        self.local = conv_bn_act(2 * d, 2 * d, 3, groups=2 * d)     # MambaVision's non-SSM branch
        self.proj = nn.Conv2d(4 * d, c, 1)
        nn.init.zeros_(self.proj.weight)                             # start as a weighted average
        nn.init.zeros_(self.proj.bias)
        self.last = {}                                               # reliability maps, for inspection

    def forward(self, fv, ft, flags=None):
        if self.gate:
            rv, rt = self.rel(fv, ft, flags)
        else:
            rv = rt = torch.ones_like(fv[:, :1])
        ft = self.align(fv, ft) if self.align is not None else ft
        pv, pt = self.phi_v(fv), self.phi_t(ft)
        yv, yt = self.mixer(pv, pt, rv, rt)
        mix = self.proj(torch.cat([self.norm(torch.cat([yv, yt], 1)),   # float32 until normalised
                                   self.local(torch.cat([pv, pt], 1))], dim=1).to(fv.dtype))
        z = (rv * fv + rt * ft) / (rv + rt + EPS) + mix
        self.last = {"rv": rv.detach(), "rt": rt.detach(), "rho": torch.maximum(rv, rt).detach()}
        return z


class ReliabilityWeightedSum(nn.Module):
    """P2 fuser: a reliability-weighted average, since a scan at stride 4 is costly and buys little."""

    def __init__(self, c: int, gate: bool = True):
        super().__init__()
        self.gate = gate
        self.rel = ReliabilityHead(c) if gate else None
        self.last = {}

    def forward(self, fv, ft, flags=None):
        if self.gate:
            rv, rt = self.rel(fv, ft, flags)
        else:
            rv = rt = torch.ones_like(fv[:, :1])
        self.last = {"rv": rv.detach(), "rt": rt.detach()}
        return (rv * fv + rt * ft) / (rv + rt + EPS)


class ConcatFusion(nn.Module):
    """The naive two-stream baseline: concatenate and squeeze back with a 1x1 conv."""

    def __init__(self, c: int):
        super().__init__()
        self.conv = conv_bn_act(2 * c, c, 1)

    def forward(self, fv, ft, flags=None):
        return self.conv(torch.cat([fv, ft], dim=1))


def build_fuser(kind: str, c: int, level: int, cfg: dict) -> nn.Module:
    """One fuser per level: concat everywhere, or a weighted sum at P2 and CMFM at P3..P5."""
    if kind == "concat":
        return ConcatFusion(c)
    gate = cfg.get("gate", True)
    if level == 2:
        return ReliabilityWeightedSum(c, gate=gate)
    return CMFM(c, ratio=cfg.get("ratio", 0.5), d_state=cfg.get("d_state", 8),
                directions=cfg.get("directions", 4), mixer=cfg.get("mixer", "ssm"), gate=gate,
                align=cfg.get("align", True), impl=cfg.get("scan_impl", "auto"),
                use_checkpoint=cfg.get("scan_checkpoint", True),
                gconv_expand=cfg.get("gconv_expand", 2.0))
