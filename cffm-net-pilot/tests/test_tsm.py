"""Temporal State Memory (Phase 2): constant memory, hold, reset, warp."""
import torch

from cffm.tsm import TemporalStateMemory, warp_state

torch.manual_seed(0)


def test_state_size_does_not_grow_with_the_stream():
    tsm = TemporalStateMemory(16, d_state=4)
    z = torch.randn(2, 16, 8, 10)
    state, shapes = None, set()
    for _ in range(25):
        _, state = tsm(z, state)
        shapes.add(tuple(state.shape))
    assert shapes == {(2, 8, 10, 8, 4)}


def test_starts_as_identity():
    tsm = TemporalStateMemory(16)
    z = torch.randn(1, 16, 6, 6)
    out, _ = tsm(z)
    assert torch.allclose(out, z)


def test_zero_reliability_holds_the_state():
    tsm = TemporalStateMemory(16)
    z = torch.randn(1, 16, 6, 6)
    _, s1 = tsm(z)
    _, s2 = tsm(torch.randn_like(z), s1, rho=torch.zeros(1, 1, 6, 6))
    assert torch.allclose(s1, s2, atol=1e-6)


def test_zero_interval_holds_the_state():
    tsm = TemporalStateMemory(16)
    z = torch.randn(1, 16, 6, 6)
    _, s1 = tsm(z)
    _, s2 = tsm(torch.randn_like(z), s1, tau_ratio=torch.zeros(1))
    assert torch.allclose(s1, s2, atol=1e-6)


def test_reset_clears_only_the_flagged_samples():
    tsm = TemporalStateMemory(16)
    z = torch.randn(2, 16, 4, 4)
    _, s = tsm(z)
    _, s_reset = tsm(z, s, reset=torch.tensor([True, False]), rho=torch.zeros(2, 1, 4, 4))
    assert s_reset[0].abs().max() == 0 and torch.allclose(s_reset[1], s[1], atol=1e-6)


def test_identity_homography_is_a_no_op_and_shift_moves_the_state():
    S = torch.randn(1, 6, 8, 3, 2)
    assert torch.allclose(warp_state(S, torch.eye(3)[None]), S, atol=1e-5)
    H = torch.tensor([[[1.0, 0, 1], [0, 1, 0], [0, 0, 1]]])                     # one cell to the right
    W = warp_state(S, H)
    assert torch.allclose(W[:, :, 1:], S[:, :, :-1], atol=1e-5)
    assert W[:, :, 0].abs().max() < 1e-5                                        # new column has no memory
