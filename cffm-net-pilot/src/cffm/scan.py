"""
The selective scan: the one piece of Mamba that actually does the "memory" work.

Given a sequence of inputs u_1..u_L, the selective SSM keeps a hidden state h
(one small vector of size N per channel) and walks along the sequence:

    h_k = exp(delta_k * A) * h_{k-1} + delta_k * B_k * u_k        (state update)
    y_k = C_k . h_k + D * u_k                                     (readout)

A is a fixed (learned) negative diagonal, so exp(delta*A) is a number in (0,1)
per state dimension: a "how much do I remember" knob. delta_k, B_k, C_k are
computed from the input itself, which is what makes the scan *selective*.
The key intuition we lean on everywhere in this project:

    delta -> 0  : exp(delta*A) -> 1 and delta*B -> 0, so the state is kept and
                  the input is ignored (a "hold" step)
    delta large : exp(delta*A) -> 0, so the past is forgotten and the state is
                  rewritten from the current input

CFFM-Net multiplies delta by a per-location sensor reliability, so a token from
an unreliable sensor simply cannot write into the state. Nothing in this file
knows about that though: here we only care about computing the recurrence fast,
correctly, and without running out of GPU memory.

Two implementations, same maths, same call signature:

  * `selective_scan_cuda`   the fused CUDA kernel from `mamba_ssm` (optional).
  * `selective_scan_torch`  plain PyTorch, a two-pass chunked scan. Runs
                            anywhere (Kaggle T4, laptop, CPU), is differentiable
                            by autograd, and is checked against a naive loop and
                            against the CUDA kernel in tests/test_scan.py.

`selective_scan` picks one automatically.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

try:  # the fused kernel only exists where mamba_ssm compiled (Linux + CUDA)
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn as _cuda_scan

    HAS_MAMBA_SSM = True
except Exception:  # ImportError on Windows or when not installed, or a broken build
    _cuda_scan = None
    HAS_MAMBA_SSM = False


def _grouped(u, delta, A, B, C, D):
    """Reshape to a grouped layout so B and C are broadcast, never copied.

    Channels are split into g groups of dg = d / g channels; every channel in a
    group shares the same B_k and C_k (that is what a group means).
        u, delta: (b, d, l)     -> (b, g, dg, l)
        A:        (d, n)        -> (g, dg, n)
        B, C:     (b, n, l) or (b, g, n, l) -> (b, g, n, l)
    """
    b, d, l = u.shape
    if B.dim() == 3:
        B, C = B.unsqueeze(1), C.unsqueeze(1)
    g, n = B.shape[1], B.shape[2]
    assert d % g == 0, f"channels ({d}) must be divisible by groups ({g})"
    dg = d // g
    return (u.view(b, g, dg, l), delta.view(b, g, dg, l), A.view(g, dg, n), B, C,
            None if D is None else D.view(g, dg))


def _scan_two_pass(u, delta, A, B, C, chunk):
    """y_k = C_k . h_k with h_k = exp(delta_k A) h_{k-1} + delta_k B_k u_k, h_0 = 0.

    u, delta: (b, g, dg, l)   A: (g, dg, n)   B, C: (b, g, n, l)   ->   y: (b, g, dg, l)

    A naive loop needs L sequential steps (10k+ at stride 8 in our model, far
    too slow in Python). Cut the sequence into K chunks of length T instead:

      pass 1  run every chunk from a zero state, all K chunks in parallel
              (T steps), keeping only each chunk's final state and the product
              of its decays;
      carry   walk across the K chunks once to get the true state entering
              each chunk;
      pass 2  run every chunk again from its true starting state (T steps),
              reading out y as we go.

    Sequential work is about 2T + K = 3 sqrt(L) steps, each a big vectorised
    op, and we never hold a (b, d, l, n) tensor: only the running state
    (b, g, dg, K, n). Decays are products of numbers in (0, 1), so nothing can
    overflow. Autograd still records each step for the backward pass; wrap the
    call in a checkpoint (use_checkpoint=True) to recompute instead of store.
    """
    b, g, dg, l = u.shape
    n = A.shape[-1]
    T = chunk
    K = math.ceil(l / T)
    pad = K * T - l
    if pad:  # padded steps have delta = 0 (keep state) and u = 0, and sit after the real sequence
        u, delta = F.pad(u, (0, pad)), F.pad(delta, (0, pad))
        B, C = F.pad(B, (0, pad)), F.pad(C, (0, pad))
    u = u.view(b, g, dg, K, T)
    delta = delta.view(b, g, dg, K, T)
    B = B.view(b, g, n, K, T).permute(0, 1, 3, 4, 2).unsqueeze(2)    # (b, g, 1, K, T, n)
    C = C.view(b, g, n, K, T).permute(0, 1, 3, 4, 2).unsqueeze(2)
    A = A[None, :, :, None, :]                                        # (1, g, dg, 1, n)

    def step(t, h):
        dt = delta[..., t].unsqueeze(-1)                              # (b, g, dg, K, 1)
        dA = torch.exp(dt * A)                                        # (b, g, dg, K, n)
        return dA, dA * h + dt * u[..., t].unsqueeze(-1) * B[..., t, :]

    # pass 1: local scans from zero, remember only the end of each chunk
    h = u.new_zeros(b, g, dg, K, n)
    P = torch.ones_like(h)
    for t in range(T):
        dA, h = step(t, h)
        P = P * dA
    # carry across chunks
    carry, starts = u.new_zeros(b, g, dg, n), []
    for k in range(K):
        starts.append(carry)
        carry = P[:, :, :, k] * carry + h[:, :, :, k]
    h = torch.stack(starts, dim=3)                                    # (b, g, dg, K, n)
    # pass 2: rerun from the true starting states and read out
    ys = []
    for t in range(T):
        _, h = step(t, h)
        ys.append((h * C[..., t, :]).sum(-1))                         # (b, g, dg, K)
    y = torch.stack(ys, dim=-1).reshape(b, g, dg, K * T)
    return y[..., :l]


def selective_scan_torch(u, delta, A, B, C, D=None, chunk: int | None = None,
                         use_checkpoint: bool = False):
    """Reference selective scan in plain PyTorch.

    u:     (b, d, l)        input sequence, d channels
    delta: (b, d, l)        step sizes, already positive (softplus applied)
    A:     (d, n)           negative real diagonal of the state matrix
    B, C:  (b, n, l) or (b, g, n, l)   input-dependent projections
    D:     (d,) or None     skip connection
    returns y: (b, d, l)

    The scan runs in float32 whatever the input dtype: the recurrence
    multiplies many numbers below 1 and half precision loses them quickly.
    `use_checkpoint` recomputes the scan in the backward pass instead of
    storing per-step tensors, trading about 30% more compute for a large
    memory saving (this is what makes batch 8 per T4 comfortable).
    """
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


def selective_scan_cuda(u, delta, A, B, C, D=None):
    """The fused kernel from mamba_ssm. Same arguments as the reference.

    delta is passed already positive, so we tell the kernel not to apply its
    own softplus (delta_softplus=False) and not to add a bias.
    """
    if not HAS_MAMBA_SSM:
        raise RuntimeError("mamba_ssm is not installed; use selective_scan_torch")
    dtype = u.dtype
    y = _cuda_scan(u.float().contiguous(), delta.float().contiguous(), A.float().contiguous(),
                   B.float().contiguous(), C.float().contiguous(),
                   None if D is None else D.float().contiguous(),
                   z=None, delta_bias=None, delta_softplus=False)
    return y.to(dtype)


def selective_scan(u, delta, A, B, C, D=None, impl: str = "auto", **kw):
    """Dispatch: 'cuda' (fused kernel), 'torch' (reference) or 'auto'.

    'auto' uses the fused kernel when it is installed and the tensors live on a
    GPU, and the PyTorch scan otherwise. Both give the same numbers up to
    float32 rounding (see tests/test_scan.py).
    """
    if impl == "cuda" or (impl == "auto" and HAS_MAMBA_SSM and u.is_cuda):
        return selective_scan_cuda(u, delta, A, B, C, D)
    return selective_scan_torch(u, delta, A, B, C, D, **kw)


def selective_scan_naive(u, delta, A, B, C, D=None):
    """The textbook O(L) loop. Slow, obviously correct; used only in tests."""
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
