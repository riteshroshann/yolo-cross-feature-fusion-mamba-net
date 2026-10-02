"""The fused Triton scan equals the textbook loop, values and gradients (GPU only)."""
import pytest
import torch

from cffm import scan

pytestmark = pytest.mark.skipif(not (scan.HAS_TRITON and torch.cuda.is_available()), reason="needs Triton and a GPU")


def _inputs(b, d, l, n, groups, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    r = lambda *s: torch.randn(*s, device="cuda", generator=g)
    u = r(b, d, l)
    delta = torch.nn.functional.softplus(r(b, d, l)) * 0.5
    A = -torch.exp(r(d, n) * 0.5)
    bc = (b, n, l) if groups is None else (b, groups, n, l)
    B, C, D = r(*bc), r(*bc), r(d)
    return [t.requires_grad_() for t in (u, delta, A, B, C, D)]


@pytest.mark.parametrize("b,d,l,n,groups", [(2, 32, 1, 8, None), (2, 40, 37, 8, 2), (1, 64, 300, 16, 4),
                                            (3, 20, 129, 4, None), (2, 256, 1000, 8, 4)])
def test_matches_naive(b, d, l, n, groups):
    xs = _inputs(b, d, l, n, groups)
    ref = scan.selective_scan_naive(*xs)
    out = scan.selective_scan_triton(*xs)
    assert torch.allclose(out, ref, atol=1e-4, rtol=1e-4), (out - ref).abs().max()
    gy = torch.randn_like(ref)
    g_ref = torch.autograd.grad(ref, xs, gy)
    g_out = torch.autograd.grad(out, xs, gy)
    for name, a, e in zip("u delta A B C D".split(), g_out, g_ref):
        err = ((a - e).abs().max() / (e.abs().max() + 1e-6)).item()
        assert err < 1e-3, f"grad {name}: relative error {err:.2e}"


def test_auto_uses_triton_on_gpu_and_matches_torch():
    xs = [t.detach() for t in _inputs(2, 64, 200, 8, 4)]
    a = scan.selective_scan(*xs, impl="auto", chunk=16, use_checkpoint=True)
    t = scan.selective_scan(*xs, impl="torch")
    assert not scan._TRITON_BROKEN and torch.allclose(a, t, atol=1e-4, rtol=1e-4)


def test_half_inputs_and_no_grad():
    xs = [t.detach().half() for t in _inputs(1, 32, 64, 8, None)]
    with torch.no_grad():
        y = scan.selective_scan_triton(*xs)
    assert y.dtype == torch.float16 and torch.isfinite(y).all()
