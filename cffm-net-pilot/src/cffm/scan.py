"""
Selective scan: h_k = exp(delta_k A) h_{k-1} + delta_k B_k u_k,  y_k = C_k . h_k + D u_k.

Implementations, same maths and signature: mamba_ssm CUDA kernel (optional), a fused Triton kernel
(any recent NVIDIA GPU, including T4), and a two-pass chunked PyTorch scan (anywhere, the reference).
"""
from __future__ import annotations

import math
import warnings

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn as _cuda_scan
    HAS_MAMBA_SSM = True
except Exception:
    _cuda_scan = None
    HAS_MAMBA_SSM = False

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except Exception:
    HAS_TRITON = False


# --------------------------------------------------------------------------- #
# PyTorch reference
# --------------------------------------------------------------------------- #
def _grouped(u, delta, A, B, C, D):
    """Split d channels into g groups that share B and C: (b, d, l) -> (b, g, dg, l)."""
    b, d, l = u.shape
    if B.dim() == 3:
        B, C = B.unsqueeze(1), C.unsqueeze(1)
    g, n = B.shape[1], B.shape[2]
    assert d % g == 0, f"channels ({d}) must be divisible by groups ({g})"
    dg = d // g
    return (u.view(b, g, dg, l), delta.view(b, g, dg, l), A.view(g, dg, n), B, C,
            None if D is None else D.view(g, dg))


def _scan_two_pass(u, delta, A, B, C, chunk):
    """Chunked scan in about 3 sqrt(L) sequential steps: local scans, carry across chunks, rerun."""
    b, g, dg, l = u.shape
    n = A.shape[-1]
    T = chunk
    K = math.ceil(l / T)
    pad = K * T - l
    if pad:  # padded steps have delta = 0, so they keep the state
        u, delta = F.pad(u, (0, pad)), F.pad(delta, (0, pad))
        B, C = F.pad(B, (0, pad)), F.pad(C, (0, pad))
    u = u.view(b, g, dg, K, T)
    delta = delta.view(b, g, dg, K, T)
    B = B.view(b, g, n, K, T).permute(0, 1, 3, 4, 2).unsqueeze(2)    # (b, g, 1, K, T, n)
    C = C.view(b, g, n, K, T).permute(0, 1, 3, 4, 2).unsqueeze(2)
    A = A[None, :, :, None, :]                                        # (1, g, dg, 1, n)

    def step(t, h):
        dt = delta[..., t].unsqueeze(-1)
        dA = torch.exp(dt * A)
        return dA, dA * h + dt * u[..., t].unsqueeze(-1) * B[..., t, :]

    h = u.new_zeros(b, g, dg, K, n)
    P = torch.ones_like(h)
    for t in range(T):
        dA, h = step(t, h)
        P = P * dA
    carry, starts = u.new_zeros(b, g, dg, n), []
    for k in range(K):
        starts.append(carry)
        carry = P[:, :, :, k] * carry + h[:, :, :, k]
    h = torch.stack(starts, dim=3)
    ys = []
    for t in range(T):
        _, h = step(t, h)
        ys.append((h * C[..., t, :]).sum(-1))
    y = torch.stack(ys, dim=-1).reshape(b, g, dg, K * T)
    return y[..., :l]


def selective_scan_torch(u, delta, A, B, C, D=None, chunk: int | None = None,
                         use_checkpoint: bool = False):
    """u, delta: (b, d, l); A: (d, n); B, C: (b, n, l) or (b, g, n, l); D: (d,). Runs in float32."""
    dtype = u.dtype
    b, d, l = u.shape
    chunk = chunk or max(16, int(math.ceil(math.sqrt(l))))
    ug, dg_, Ag, Bg, Cg, Dg = _grouped(u.float(), delta.float(), A.float(), B.float(), C.float(),
                                       None if D is None else D.float())

    def run(ug, dg_, Ag, Bg, Cg):
        return _scan_two_pass(ug, dg_, Ag, Bg, Cg, chunk)

    if use_checkpoint and torch.is_grad_enabled():
        y = checkpoint(run, ug, dg_, Ag, Bg, Cg, use_reentrant=False)
    else:
        y = run(ug, dg_, Ag, Bg, Cg)
    if Dg is not None:
        y = y + Dg[None, :, :, None] * ug
    return y.reshape(b, d, l).to(dtype)


# --------------------------------------------------------------------------- #
# fused Triton kernel: one program per (batch, group, block of channels) walks the sequence
# --------------------------------------------------------------------------- #
if HAS_TRITON:
    @triton.jit
    def _fwd_kernel(u_ptr, dt_ptr, A_ptr, B_ptr, C_ptr, y_ptr, h_ptr, L, D, G, DG,
                    N: tl.constexpr, NP: tl.constexpr, BD: tl.constexpr, STORE_H: tl.constexpr):
        pid_bg, pid_d = tl.program_id(0), tl.program_id(1)
        bi, gi = (pid_bg // G).to(tl.int64), pid_bg % G
        od = pid_d * BD + tl.arange(0, BD)
        on = tl.arange(0, NP)
        md, mn = od < DG, on < N
        ch = gi * DG + od
        m2 = md[:, None] & mn[None, :]
        A = tl.load(A_ptr + ch[:, None] * N + on[None, :], mask=m2, other=0.0)
        x_off = bi * L * D + ch                       # u, delta, y: (b, L, D)
        bc_off = (bi * G + gi) * L * N + on           # B, C: (b, G, L, N)
        h_off = bi * L * D * N + ch[:, None] * N + on[None, :]   # h: (b, L, D, N)
        h = tl.zeros((BD, NP), dtype=tl.float32)
        for t in range(L):
            u = tl.load(u_ptr + x_off + t * D, mask=md, other=0.0)
            dt = tl.load(dt_ptr + x_off + t * D, mask=md, other=0.0)
            b_t = tl.load(B_ptr + bc_off + t * N, mask=mn, other=0.0)
            c_t = tl.load(C_ptr + bc_off + t * N, mask=mn, other=0.0)
            h = tl.exp(dt[:, None] * A) * h + (dt * u)[:, None] * b_t[None, :]
            tl.store(y_ptr + x_off + t * D, tl.sum(h * c_t[None, :], axis=1), mask=md)
            if STORE_H:
                tl.store(h_ptr + h_off + t * D * N, h, mask=m2)

    @triton.jit
    def _bwd_kernel(u_ptr, dt_ptr, A_ptr, B_ptr, C_ptr, h_ptr, dy_ptr,
                    du_ptr, ddt_ptr, dA_ptr, dB_ptr, dC_ptr, L, D, G, DG, NBLK,
                    N: tl.constexpr, NP: tl.constexpr, BD: tl.constexpr):
        pid_bg, pid_d = tl.program_id(0), tl.program_id(1)
        bi, gi = (pid_bg // G).to(tl.int64), pid_bg % G
        od = pid_d * BD + tl.arange(0, BD)
        on = tl.arange(0, NP)
        md, mn = od < DG, on < N
        ch = gi * DG + od
        m2 = md[:, None] & mn[None, :]
        A = tl.load(A_ptr + ch[:, None] * N + on[None, :], mask=m2, other=0.0)
        x_off = bi * L * D + ch
        bc_off = (bi * G + gi) * L * N + on
        h_off = bi * L * D * N + ch[:, None] * N + on[None, :]
        part_off = ((bi * G + gi) * NBLK + pid_d) * L * N + on   # per-block partial dB, dC
        dh = tl.zeros((BD, NP), dtype=tl.float32)
        dA_next = tl.zeros((BD, NP), dtype=tl.float32)
        dA_acc = tl.zeros((BD, NP), dtype=tl.float32)
        for i in range(L):
            t = L - 1 - i
            u = tl.load(u_ptr + x_off + t * D, mask=md, other=0.0)
            dt = tl.load(dt_ptr + x_off + t * D, mask=md, other=0.0)
            g = tl.load(dy_ptr + x_off + t * D, mask=md, other=0.0)
            b_t = tl.load(B_ptr + bc_off + t * N, mask=mn, other=0.0)
            c_t = tl.load(C_ptr + bc_off + t * N, mask=mn, other=0.0)
            h_t = tl.load(h_ptr + h_off + t * D * N, mask=m2, other=0.0)
            h_prev = tl.load(h_ptr + h_off + (t - 1) * D * N, mask=m2 & (t > 0), other=0.0)
            dA = tl.exp(dt[:, None] * A)
            dh = dh * dA_next + g[:, None] * c_t[None, :]           # dL/dh_t
            tl.store(dC_ptr + part_off + t * N, tl.sum(g[:, None] * h_t, axis=0), mask=mn)
            tl.store(dB_ptr + part_off + t * N, tl.sum(dh * (dt * u)[:, None], axis=0), mask=mn)
            hd = dh * h_prev * dA                                   # through exp(delta A)
            dhb = tl.sum(dh * b_t[None, :], axis=1)
            tl.store(ddt_ptr + x_off + t * D, dhb * u + tl.sum(hd * A, axis=1), mask=md)
            tl.store(du_ptr + x_off + t * D, dhb * dt, mask=md)
            dA_acc += hd * dt[:, None]
            dA_next = dA
        tl.store(dA_ptr + bi * D * N + ch[:, None] * N + on[None, :], dA_acc, mask=m2)


class _TritonScan(torch.autograd.Function):
    BD = 16

    @staticmethod
    def forward(ctx, u, delta, A, B, C, store):
        b, D, L = u.shape
        G, N = B.shape[1], B.shape[2]
        DG = D // G
        ut, dtt = u.transpose(1, 2).contiguous(), delta.transpose(1, 2).contiguous()
        Bt, Ct = B.transpose(2, 3).contiguous(), C.transpose(2, 3).contiguous()
        A = A.contiguous()
        y = torch.empty_like(ut)
        H = torch.empty(b, L, D, N, device=u.device, dtype=torch.float32) if store else y
        grid = (b * G, triton.cdiv(DG, _TritonScan.BD))
        _fwd_kernel[grid](ut, dtt, A, Bt, Ct, y, H, L, D, G, DG, N=N, NP=triton.next_power_of_2(N),
                          BD=_TritonScan.BD, STORE_H=store, num_warps=1)
        if store:
            ctx.save_for_backward(ut, dtt, A, Bt, Ct, H)
        return y.transpose(1, 2)

    @staticmethod
    def backward(ctx, dy):
        ut, dtt, A, Bt, Ct, H = ctx.saved_tensors
        b, L, D = ut.shape
        G, N = Bt.shape[1], Bt.shape[3]
        DG = D // G
        nblk = triton.cdiv(DG, _TritonScan.BD)
        dyt = dy.transpose(1, 2).contiguous().float()
        du, ddt = torch.empty_like(ut), torch.empty_like(ut)
        dA = torch.empty(b, D, N, device=ut.device, dtype=torch.float32)
        dB = torch.empty(b, G, nblk, L, N, device=ut.device, dtype=torch.float32)
        dC = torch.empty_like(dB)
        _bwd_kernel[(b * G, nblk)](ut, dtt, A, Bt, Ct, H, dyt, du, ddt, dA, dB, dC, L, D, G, DG, nblk,
                                   N=N, NP=triton.next_power_of_2(N), BD=_TritonScan.BD, num_warps=1)
        return (du.transpose(1, 2), ddt.transpose(1, 2), dA.sum(0),
                dB.sum(2).transpose(2, 3), dC.sum(2).transpose(2, 3), None)


def selective_scan_triton(u, delta, A, B, C, D=None):
    """Fused Triton scan, float32, same arguments as the reference."""
    dtype = u.dtype
    squeeze = B.dim() == 3
    if squeeze:
        B, C = B.unsqueeze(1), C.unsqueeze(1)
    u32 = u.float()
    store = torch.is_grad_enabled() and any(t.requires_grad for t in (u, delta, A, B, C))
    y = _TritonScan.apply(u32, delta.float(), A.float(), B.float(), C.float(), store)
    if D is not None:
        y = y + D.float()[None, :, None] * u32
    return y.to(dtype)


# --------------------------------------------------------------------------- #
# mamba_ssm kernel and dispatch
# --------------------------------------------------------------------------- #
def selective_scan_cuda(u, delta, A, B, C, D=None):
    """mamba_ssm's fused kernel; delta is already positive, so no softplus inside."""
    if not HAS_MAMBA_SSM:
        raise RuntimeError("mamba_ssm is not installed; use selective_scan_torch")
    dtype = u.dtype
    y = _cuda_scan(u.float().contiguous(), delta.float().contiguous(), A.float().contiguous(),
                   B.float().contiguous(), C.float().contiguous(),
                   None if D is None else D.float().contiguous(),
                   z=None, delta_bias=None, delta_softplus=False)
    return y.to(dtype)


_TRITON_BROKEN = False


def selective_scan(u, delta, A, B, C, D=None, impl: str = "auto", **kw):
    """impl: 'cuda' (mamba_ssm), 'triton', 'torch', or 'auto' (the fastest available on this device)."""
    global _TRITON_BROKEN
    if impl == "cuda" or (impl == "auto" and HAS_MAMBA_SSM and u.is_cuda):
        return selective_scan_cuda(u, delta, A, B, C, D)
    if impl == "triton" or (impl == "auto" and HAS_TRITON and u.is_cuda and not _TRITON_BROKEN):
        try:
            return selective_scan_triton(u, delta, A, B, C, D)
        except Exception as e:
            if impl == "triton":
                raise
            _TRITON_BROKEN = True
            warnings.warn(f"Triton scan failed ({e}); falling back to the PyTorch scan")
    return selective_scan_torch(u, delta, A, B, C, D, **kw)


def selective_scan_naive(u, delta, A, B, C, D=None):
    """The textbook loop, for tests."""
    u, delta, A = u.float(), delta.float(), A.float()
    b, d, l = u.shape
    if B.dim() == 3:
        Bx, Cx = B.float().unsqueeze(1).expand(-1, d, -1, -1), C.float().unsqueeze(1).expand(-1, d, -1, -1)
    else:
        rep = d // B.shape[1]
        Bx, Cx = B.float().repeat_interleave(rep, 1), C.float().repeat_interleave(rep, 1)
    h = u.new_zeros(b, d, A.shape[1])
    ys = []
    for k in range(l):
        h = torch.exp(delta[:, :, k, None] * A) * h + (delta[:, :, k] * u[:, :, k])[..., None] * Bx[:, :, :, k]
        ys.append((h * Cx[:, :, :, k]).sum(-1))
    y = torch.stack(ys, dim=-1)
    if D is not None:
        y = y + D.float()[None, :, None] * u
    return y
