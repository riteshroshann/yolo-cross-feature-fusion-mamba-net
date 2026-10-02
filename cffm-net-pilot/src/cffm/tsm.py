"""
Temporal State Memory (TSM): the Phase 2 module, built and tested now, not used in the pilot.

The pilot trains the image-mode network. The streaming network adds this block
after each CMFM at P3..P5. It keeps a small state per pixel that survives from
frame to frame:

    S_t(p) = exp(delta_t(p) * A) * warp(S_{t-1}, H)(p) + delta_t(p) * u_t(p) B_t(p)^T
    y_t(p) = S_t(p) C_t(p) + D * u_t(p)
    delta_t(p) = softplus(w . u_t(p) + b) * (tau_t / tau_0) * rho_t(p)

Three things control the step, exactly as in Section 6.5 of the dossier:
  * rho_t   the reliability from CMFM: where no sensor can be trusted, the
            step goes to 0 and the memory of an object is held
  * tau_t   the real time since the previous frame: dropped frames make the
            step bigger, the way a continuous-time system would evolve
  * reset   a scene cut or a failed motion estimate clears the state

Memory per frame is O(H * W * C_s * N), whatever the length of the stream.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def warp_state(S: torch.Tensor, H: torch.Tensor | None) -> torch.Tensor:
    """Warp a (b, h, w, c, n) state with per-sample 3x3 homographies (b, 3, 3).

    H maps pixel coordinates of frame t-1 to frame t at this pyramid level.
    We sample S_{t-1} at H^{-1}(p) for every p of frame t; pixels coming from
    outside the old view get a zero state (nothing is remembered there).
    """
    if H is None:
        return S
    b, h, w, c, n = S.shape
    ys, xs = torch.meshgrid(torch.arange(h, device=S.device, dtype=S.dtype),
                            torch.arange(w, device=S.device, dtype=S.dtype), indexing="ij")
    pts = torch.stack([xs, ys, torch.ones_like(xs)], -1).reshape(1, -1, 3)         # (1, hw, 3)
    src = pts @ torch.linalg.inv(H.to(S.dtype)).transpose(1, 2)                     # (b, hw, 3)
    src = src[..., :2] / src[..., 2:].clamp_min(1e-6)
    gx = src[..., 0] / max(w - 1, 1) * 2 - 1
    gy = src[..., 1] / max(h - 1, 1) * 2 - 1
    grid = torch.stack([gx, gy], -1).view(b, h, w, 2)
    flat = S.permute(0, 3, 4, 1, 2).reshape(b, c * n, h, w)
    out = F.grid_sample(flat, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
    return out.view(b, c, n, h, w).permute(0, 3, 4, 1, 2)


class TemporalStateMemory(nn.Module):
    """One TSM block for one pyramid level with c channels."""

    def __init__(self, c: int, ratio: float = 0.5, d_state: int = 8, dt_min=1e-3, dt_max=1e-1):
        super().__init__()
        cs = max(8, int(round(c * ratio)))
        self.cs, self.n = cs, d_state
        self.inp = nn.Conv2d(c, cs, 1)
        self.to_bc = nn.Conv2d(cs, 2 * d_state, 1)
        self.to_dt = nn.Conv2d(cs, cs, 1)
        dt = torch.exp(torch.rand(cs) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min))
        with torch.no_grad():
            self.to_dt.bias.copy_(dt + torch.log(-torch.expm1(-dt)))
        self.A_log = nn.Parameter(torch.log(torch.arange(1, d_state + 1, dtype=torch.float32)).repeat(cs, 1))
        self.D = nn.Parameter(torch.ones(cs))
        self.spread = nn.Conv2d(cs, cs, 3, padding=1, groups=cs)          # spread evidence to neighbours
        self.out = nn.Conv2d(cs, c, 1)
        nn.init.zeros_(self.out.weight)                                    # starts as the identity
        nn.init.zeros_(self.out.bias)

    def init_state(self, z: torch.Tensor) -> torch.Tensor:
        b, _, h, w = z.shape
        return z.new_zeros(b, h, w, self.cs, self.n)

    def forward(self, z, state=None, rho=None, tau_ratio=None, H=None, reset=None):
        """One frame. z: (b, c, h, w). Returns (z_out, new_state).

        rho:       (b, 1, h, w) reliability in [0, 1], default 1
        tau_ratio: (b,) real interval / training interval, default 1
        H:         (b, 3, 3) homography from the previous frame, default identity
        reset:     (b,) bool, clear the state for these samples
        """
        b, _, h, w = z.shape
        u = self.inp(z)                                                    # (b, cs, h, w)
        if state is None:
            state = self.init_state(z)
        state = warp_state(state, H)
        if reset is not None:
            state = state * (~reset).to(state.dtype).view(b, 1, 1, 1, 1)
        delta = F.softplus(self.to_dt(u))                                  # (b, cs, h, w)
        if tau_ratio is not None:
            delta = delta * tau_ratio.to(delta.dtype).view(b, 1, 1, 1)
        if rho is not None:
            delta = delta * rho
        Bc, Cc = self.to_bc(u).chunk(2, dim=1)                             # (b, n, h, w) each
        A = -torch.exp(self.A_log.float())                                 # (cs, n)
        d_ = delta.permute(0, 2, 3, 1).unsqueeze(-1).float()               # (b, h, w, cs, 1)
        u_ = u.permute(0, 2, 3, 1).unsqueeze(-1).float()
        B_ = Bc.permute(0, 2, 3, 1).unsqueeze(-2).float()                  # (b, h, w, 1, n)
        C_ = Cc.permute(0, 2, 3, 1).unsqueeze(-2).float()
        state = torch.exp(d_ * A) * state.float() + d_ * u_ * B_           # (b, h, w, cs, n)
        y = (state * C_).sum(-1) + self.D * u.permute(0, 2, 3, 1).float()  # (b, h, w, cs)
        y = y.permute(0, 3, 1, 2).to(z.dtype)
        return z + self.out(self.spread(y)), state
