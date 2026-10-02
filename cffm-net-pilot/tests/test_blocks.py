"""Fusion blocks: shapes, starting behaviour, and the gating property the method relies on."""
import pytest
import torch

from cffm.blocks import (CMFM, ConcatFusion, GatedConvMixer, GatedCrossScan, OffsetAlign, ReliabilityHead,
                         ReliabilityWeightedSum, build_fuser)

torch.manual_seed(0)


def feats(b=2, c=32, h=10, w=12):
    return torch.randn(b, c, h, w), torch.randn(b, c, h, w)


def test_reliability_starts_high_and_respects_flags():
    head = ReliabilityHead(32).eval()
    fv, ft = feats()
    rv, rt = head(fv, ft)
    assert rv.shape == (2, 1, 10, 12)
    assert torch.allclose(rv, torch.full_like(rv, torch.sigmoid(torch.tensor(2.0)).item()), atol=1e-6)
    rv, rt = head(fv, ft, flags=torch.tensor([[1.0, 0.0], [0.0, 1.0]]))
    assert rt[0].abs().max() == 0 and rv[1].abs().max() == 0


def test_offset_align_starts_as_identity():
    al = OffsetAlign(32).eval()
    fv, ft = feats()
    assert torch.allclose(al(fv, ft), ft, atol=1e-5)


@pytest.mark.parametrize("K", [2, 4])
def test_scan_orders_are_invertible(K):
    scan = GatedCrossScan(8, directions=K)
    X = torch.randn(2, 8, 5, 7, 2)
    seqs = scan._orders(X)
    assert seqs.shape == (2, K, 8, 2 * 5 * 7)
    assert torch.allclose(scan._merge(seqs, 5, 7), K * X)   # every direction maps back to X


def test_cross_scan_interleaves_modalities():
    scan = GatedCrossScan(4, directions=2)
    xv, xt = torch.zeros(1, 4, 2, 2), torch.ones(1, 4, 2, 2)
    seq = scan._orders(torch.stack([xv, xt], -1))[0, 0, 0]  # direction 0, channel 0
    assert seq.tolist() == [0, 1, 0, 1, 0, 1, 0, 1]


def test_zero_reliability_thermal_cannot_change_visible_outputs():
    """With r_t = 0 every thermal token gets delta = 0 and is skipped by the state."""
    scan = GatedCrossScan(8, directions=4, impl="torch").eval()
    xv, xt = torch.randn(1, 8, 6, 6), torch.randn(1, 8, 6, 6)
    rv, rt = torch.ones(1, 1, 6, 6), torch.zeros(1, 1, 6, 6)
    yv1, _ = scan(xv, xt, rv, rt)
    yv2, _ = scan(xv, xt + 3 * torch.randn_like(xt), rv, rt)
    assert torch.allclose(yv1, yv2, atol=1e-5)
    # with r_t = 1 thermal does reach the visible outputs
    yv3, _ = scan(xv, xt, rv, torch.ones_like(rt))
    yv4, _ = scan(xv, xt + 3 * torch.randn_like(xt), rv, torch.ones_like(rt))
    assert not torch.allclose(yv3, yv4, atol=1e-3)


@pytest.mark.parametrize("mixer", ["ssm", "gconv"])
@pytest.mark.parametrize("gate", [True, False])
def test_cmfm_starts_as_a_weighted_average(mixer, gate):
    blk = CMFM(32, mixer=mixer, gate=gate, impl="torch").eval()
    fv, ft = feats()
    z = blk(fv, ft)
    assert z.shape == fv.shape
    assert torch.allclose(z, (fv + ft) / 2, atol=1e-4)      # zero-init projection, equal reliabilities


def test_cmfm_backward():
    blk = CMFM(16, impl="torch")
    fv, ft = (t.requires_grad_(True) for t in feats(c=16, h=8, w=8))
    blk(fv, ft).square().mean().backward()
    assert fv.grad is not None and ft.grad is not None
    assert all(p.grad is not None for n, p in blk.named_parameters() if p.requires_grad and "proj" in n)


def test_other_fusers_shapes():
    fv, ft = feats()
    assert ConcatFusion(32)(fv, ft).shape == fv.shape
    assert ReliabilityWeightedSum(32)(fv, ft).shape == fv.shape
    yv, yt = GatedConvMixer(16)(torch.randn(2, 16, 5, 5), torch.randn(2, 16, 5, 5),
                                torch.ones(2, 1, 5, 5), torch.ones(2, 1, 5, 5))
    assert yv.shape == yt.shape == (2, 16, 5, 5)


def test_build_fuser_by_level():
    assert isinstance(build_fuser("cffm", 32, 2, {}), ReliabilityWeightedSum)
    assert isinstance(build_fuser("cffm", 32, 3, {}), CMFM)
    assert isinstance(build_fuser("concat", 32, 3, {}), ConcatFusion)
    assert isinstance(build_fuser("cffm", 32, 4, {"mixer": "gconv"}).mixer, GatedConvMixer)
