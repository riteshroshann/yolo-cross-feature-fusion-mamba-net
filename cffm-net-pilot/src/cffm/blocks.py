"""
Fusion blocks of CFFM-Net.

The story of one fusion step, at one pyramid level, in plain words:

  1. Look at the visible and thermal feature maps side by side and decide, at
     every pixel, how much to trust each one (ReliabilityHead).
  2. Nudge the thermal features so they line up with the visible ones; real
     camera pairs are never perfectly registered (OffsetAlign).
  3. Interleave the two modalities into one token sequence, visible then
     thermal at every location, and run a selective scan over it in four
     directions. The step size of every token is multiplied by that token's
     reliability, so an untrustworthy token can neither write into the state
     nor wipe it (GatedCrossScan).
  4. Add a cheap local convolution branch, project back, and add a
     reliability-weighted average of the inputs as the residual (CMFM).

Everything is shape-annotated: b = batch, c = channels of the backbone level,
d = channels inside the block, h, w = spatial size, L = sequence length.

Also here, so the ablations share every line of code they can:
  * ReliabilityWeightedSum   the cheap fuser used at stride 4 (P2)
  * ConcatFusion             the naive two-stream baseline
  * GatedConvMixer           the MambaOut-style control: same block, no SSM
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .scan import selective_scan

EPS = 1e-6


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
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


# --------------------------------------------------------------------------- #
# 1. how much do we trust each sensor, here?
# --------------------------------------------------------------------------- #
class ReliabilityHead(nn.Module):
    """Per-pixel reliability r in (0, 1) for each modality.

    Input is [F_v, F_t, |F_v - F_t|]: the two views plus where they disagree,
    because disagreement is the most direct hint that one of them is degraded.
    The last conv is zero-initialised with bias +2, so at the start of training
    r = sigmoid(2) = 0.88 everywhere: trust both, then learn when not to.

    `flags` (b, 2) are hard sensor-availability bits from the outside world
    (1 = frame arrived, 0 = sensor missing). They simply multiply r, so a known
    outage needs no learning at all.
    """

    def __init__(self, c: int, init_bias: float = 2.0):
        super().__init__()
        self.body = conv_bn_act(3 * c, 3 * c, k=3, groups=3 * c)    # depthwise, cheap
        self.out = nn.Conv2d(3 * c, 2, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.constant_(self.out.bias, init_bias)

    def forward(self, fv, ft, flags=None):
        x = torch.cat([fv, ft, (fv - ft).abs()], dim=1)              # (b, 3c, h, w)
        r = torch.sigmoid(self.out(self.body(x)))                    # (b, 2, h, w)
        if flags is not None:
            r = r * flags.to(device=r.device, dtype=r.dtype)[:, :, None, None]
        return r[:, :1], r[:, 1:]                                    # (b, 1, h, w) each


# --------------------------------------------------------------------------- #
# 2. line the thermal features up with the visible ones
# --------------------------------------------------------------------------- #
class OffsetAlign(nn.Module):
    """Predict a small displacement field and resample thermal onto visible.

    The last conv is zero-initialised and the output goes through tanh, so the
    block starts as the identity and can never move a feature by more than
    `max_disp` cells of this pyramid level. This follows the offset guidance of
    COMO, applied here before the scan.
    """

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
        grid = torch.stack([gx, gy], dim=-1).unsqueeze(0)            # (1, h, w, 2), identity grid
        scale = torch.tensor([2.0 / max(w - 1, 1), 2.0 / max(h - 1, 1)], device=ft.device, dtype=ft.dtype)
        grid = grid + off.permute(0, 2, 3, 1) * scale                # cells -> normalised coords
        return F.grid_sample(ft, grid, mode="bilinear", padding_mode="border", align_corners=True)


# --------------------------------------------------------------------------- #
# 3. the heart of it: interleaved cross-modal scan with reliability-gated steps
# --------------------------------------------------------------------------- #
class GatedCrossScan(nn.Module):
    """Selective scan over interleaved visible/thermal tokens, K directions.

    Token order (direction 0) for an h x w map is

        v(0,0) t(0,0) v(0,1) t(0,1) ... v(h-1,w-1) t(h-1,w-1)

    so the hidden state hops visible -> thermal -> visible at every location:
    fusion happens *inside* the state, not after it. Directions 1..3 are the
    reverse, column-major and reverse column-major orders (VMamba's SS2D), each
    with its own parameters; outputs are mapped back and summed.

    The only change to a standard Mamba scan is one line:

        delta = softplus(dt_proj(x) + bias) * r          # r = token reliability

    With r -> 0 the step goes to 0, so exp(delta*A) -> 1 and delta*B -> 0: the
    token is skipped and the state carries on untouched.
    """

    def __init__(self, d: int, d_state: int = 8, directions: int = 4, dt_rank: int | None = None,
                 dt_min: float = 1e-3, dt_max: float = 1e-1, impl: str = "auto",
                 chunk: int | None = None, use_checkpoint: bool = True):
        super().__init__()
        assert directions in (2, 4), "use 4 (rows and columns, both ways) or 2 (rows and columns, forward)"
        self.d, self.n, self.K = d, d_state, directions
        self.r = dt_rank or math.ceil(d / 16)
        self.impl, self.chunk, self.use_checkpoint = impl, chunk, use_checkpoint
        K, r, n = self.K, self.r, self.n

        # x -> (dt, B, C), one projection per direction
        bound = 1.0 / math.sqrt(d)
        self.x_proj = nn.Parameter(torch.empty(K, r + 2 * n, d).uniform_(-bound, bound))
        # dt -> per-channel step, initialised so softplus(bias) lies in [dt_min, dt_max]
        self.dt_w = nn.Parameter(torch.empty(K, d, r).uniform_(-r ** -0.5, r ** -0.5))
        dt = torch.exp(torch.rand(K, d) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min))
        self.dt_b = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))  # inverse softplus
        # A = -exp(A_log) with the usual 1..n initialisation; D = skip
        self.A_log = nn.Parameter(torch.log(torch.arange(1, n + 1, dtype=torch.float32)).repeat(K * d, 1))
        self.D = nn.Parameter(torch.ones(K * d))

    # -- ordering helpers --------------------------------------------------- #
    def _orders(self, X):
        """(b, c, h, w, 2) -> (b, K, c, L) with L = 2hw, in the K scan orders."""
        b, c = X.shape[:2]
        rows = X.reshape(b, c, -1)                                   # (h, w, m) flattened
        cols = X.permute(0, 1, 3, 2, 4).reshape(b, c, -1)            # (w, h, m) flattened
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
        # xv, xt: (b, d, h, w)    rv, rt: (b, 1, h, w)
        b, d, h, w = xv.shape
        K, n, r = self.K, self.n, self.r
        seqs = self._orders(torch.stack([xv, xt], dim=-1))           # (b, K, d, L)
        rel = self._orders(torch.stack([rv, rt], dim=-1))            # (b, K, 1, L)
        L = seqs.shape[-1]

        x_dbl = torch.einsum("bkdl,kcd->bkcl", seqs, self.x_proj)   # (b, K, r+2n, L)
        dts, Bs, Cs = torch.split(x_dbl, [r, n, n], dim=2)
        dts = torch.einsum("bkrl,kdr->bkdl", dts, self.dt_w) + self.dt_b[None, :, :, None]
        delta = F.softplus(dts) * rel                                # <- the reliability gate

        A = -torch.exp(self.A_log.float())                           # (K*d, n)
        y = selective_scan(seqs.reshape(b, K * d, L), delta.reshape(b, K * d, L), A,
                           Bs.contiguous(), Cs.contiguous(), self.D, impl=self.impl,
                           **({} if self.impl == "cuda" else
                              {"chunk": self.chunk, "use_checkpoint": self.use_checkpoint}))
        Y = self._merge(y.view(b, K, d, L), h, w)                    # (b, d, h, w, 2)
        return Y[..., 0], Y[..., 1]


class GatedConvMixer(nn.Module):
    """The control for contribution C4: same interface, no state-space model.

    Reliability gates the *input* (there is no step size to gate), then a
    gated depthwise convolution mixes space and a 1x1 conv mixes modalities.
    `expand` widens the hidden layer so its cost can be matched to the scan;
    notebook 08 chooses it by measuring GFLOPs of both.
    """

    def __init__(self, d: int, kernel: int = 7, expand: float = 2.0):
        super().__init__()
        hid = int(2 * d * expand)
        self.inp = nn.Conv2d(2 * d, hid, 1)
        self.dw = nn.Conv2d(hid, hid, kernel, padding=kernel // 2, groups=hid)
        self.gate = nn.Conv2d(2 * d, hid, 1)
        self.out = nn.Conv2d(hid, 2 * d, 1)

    def forward(self, xv, xt, rv, rt):
        x = torch.cat([xv * rv, xt * rt], dim=1)                     # (b, 2d, h, w)
        y = self.out(self.dw(self.inp(x)) * torch.sigmoid(self.gate(x)))
        return y.chunk(2, dim=1)


# --------------------------------------------------------------------------- #
# 4. the full Cross-Modal Fusion Mamba block
# --------------------------------------------------------------------------- #
class CMFM(nn.Module):
    """Cross-Modal Fusion Mamba block: (F_v, F_t) -> Z, all (b, c, h, w).

    Arguments that matter for the ablations:
      mixer  'ssm'   the reliability-gated cross scan (CFFM-Net)
             'gconv' the gated-convolution control (C4)
      gate   False forces r = 1 everywhere (gating-off ablation, C1)
      align  False skips the offset alignment
    """

    def __init__(self, c: int, ratio: float = 0.5, d_state: int = 8, directions: int = 4,
                 mixer: str = "ssm", gate: bool = True, align: bool = True, impl: str = "auto",
                 use_checkpoint: bool = True, gconv_expand: float = 2.0):
        super().__init__()
        d = max(16, int(round(c * ratio)))
        self.gate, self.c, self.d = gate, c, d
        self.rel = ReliabilityHead(c) if gate else None   # no unused parameters when gating is off
        self.align = OffsetAlign(c) if align else None
        # per-modality projection into the block: 1x1 conv then a 3x3 depthwise
        # conv for local context (the role Mamba's short causal conv plays in 1D)
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
        pv, pt = self.phi_v(fv), self.phi_t(ft)                      # (b, d, h, w)
        yv, yt = self.mixer(pv, pt, rv, rt)                          # (b, d, h, w)
        mix = self.proj(torch.cat([self.norm(torch.cat([yv, yt], 1)),
                                   self.local(torch.cat([pv, pt], 1))], dim=1))
        z = (rv * fv + rt * ft) / (rv + rt + EPS) + mix
        self.last = {"rv": rv.detach(), "rt": rt.detach(), "rho": torch.maximum(rv, rt).detach()}
        return z


class ReliabilityWeightedSum(nn.Module):
    """Stride-4 fuser: at 160 x 128 a scan would be expensive and buys little,
    so P2 gets a reliability-weighted average and nothing more."""

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
    """One fuser for one pyramid level.

    kind 'concat' gives the baseline everywhere. Otherwise P2 gets the
    reliability-weighted sum and P3..P5 get CMFM with the configured mixer.
    """
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
