"""The selective scan must be exactly the recurrence it claims to be."""
import pytest
import torch

from cffm.scan import HAS_MAMBA_SSM, selective_scan_cuda, selective_scan_naive, selective_scan_torch

torch.manual_seed(0)


def rand_inputs(b=2, d=6, l=37, n=4, groups=None, device="cpu"):
    u = torch.randn(b, d, l, device=device)
    delta = torch.nn.functional.softplus(torch.randn(b, d, l, device=device))
    A = -torch.exp(torch.randn(d, n, device=device))
    shape = (b, groups, n, l) if groups else (b, n, l)
    B, C = torch.randn(*shape, device=device), torch.randn(*shape, device=device)
    D = torch.randn(d, device=device)
    return u, delta, A, B, C, D


@pytest.mark.parametrize("l", [1, 7, 37, 256, 1000])
@pytest.mark.parametrize("groups", [None, 2])
def test_chunked_matches_naive(l, groups):
    args = rand_inputs(l=l, groups=groups)
    ref = selective_scan_naive(*args)
    for chunk in (None, 5, 64):
        out = selective_scan_torch(*args, chunk=chunk)
        assert torch.allclose(out, ref, atol=1e-4, rtol=1e-4), f"chunk={chunk}"


def test_gradients_match_naive():
    args = [t.clone().requires_grad_(True) for t in rand_inputs(l=50)]
    g1 = torch.autograd.grad(selective_scan_naive(*args).square().sum(), args)
    g2 = torch.autograd.grad(selective_scan_torch(*args, chunk=8).square().sum(), args)
    for a, b in zip(g1, g2):
        assert torch.allclose(a, b, atol=1e-3, rtol=1e-3)


def test_checkpoint_gives_same_result_and_grad():
    args = [t.clone().requires_grad_(True) for t in rand_inputs(l=64)]
    y1 = selective_scan_torch(*args, use_checkpoint=False)
    y2 = selective_scan_torch(*args, use_checkpoint=True)
    assert torch.allclose(y1, y2)
    g1 = torch.autograd.grad(y1.sum(), args[0], retain_graph=True)[0]
    g2 = torch.autograd.grad(y2.sum(), args[0])[0]
    assert torch.allclose(g1, g2, atol=1e-5)


def test_zero_step_tokens_cannot_write_into_the_state():
    """delta = 0 from position j on: later inputs may only reach y through the D skip."""
    u, delta, A, B, C, D = rand_inputs(l=20)
    j = 12
    delta[:, :, j:] = 0
    y1 = selective_scan_torch(u, delta, A, B, C, D)
    u2 = u.clone()
    u2[:, :, j:] += 5 * torch.randn_like(u2[:, :, j:])
    y2 = selective_scan_torch(u2, delta, A, B, C, D)
    expected = D[None, :, None] * (u2 - u)
    assert torch.allclose((y2 - y1)[:, :, j:], expected[:, :, j:], atol=1e-4)
    assert torch.allclose((y2 - y1)[:, :, :j], torch.zeros_like(y1[:, :, :j]), atol=1e-6)


@pytest.mark.skipif(not (HAS_MAMBA_SSM and torch.cuda.is_available()), reason="mamba_ssm CUDA kernel not available")
@pytest.mark.parametrize("groups", [None, 4])
def test_cuda_kernel_matches_reference(groups):
    args = rand_inputs(b=2, d=16, l=513, n=8, groups=groups, device="cuda")
    ref = selective_scan_torch(*args)
    out = selective_scan_cuda(*args)
    assert torch.allclose(out, ref, atol=1e-3, rtol=1e-3)
