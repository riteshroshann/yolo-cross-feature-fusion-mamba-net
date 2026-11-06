import pytest
import torch

from cffm import faults


@pytest.fixture
def x():
    return torch.rand(2, 4, 32, 40, generator=torch.Generator().manual_seed(0))


@pytest.mark.parametrize("fault", faults.FAULTS)
@pytest.mark.parametrize("severity", faults.SEVERITIES)
def test_fault_touches_only_its_sensor(x, fault, severity):
    y = faults.apply(x, fault, severity)
    other = slice(3, None) if fault.startswith("visible") else slice(0, 3)
    own = slice(0, 3) if fault.startswith("visible") else slice(3, None)
    assert y.shape == x.shape and y.min() >= 0 and y.max() <= 1
    assert torch.equal(y[:, other], x[:, other])
    assert not torch.equal(y[:, own], x[:, own])


def test_deterministic_and_input_untouched(x):
    before = x.clone()
    assert torch.equal(faults.apply(x, "visible_noise", 3), faults.apply(x, "visible_noise", 3))
    assert torch.equal(x, before)


@pytest.mark.parametrize("fault", ["visible_dark", "visible_blur", "thermal_noise", "thermal_crossover"])
def test_severity_is_monotone(x, fault):
    err = [(faults.apply(x, fault, s) - x).abs().mean() for s in faults.SEVERITIES]
    assert err[0] < err[1] < err[2]


def test_unknown_fault_raises(x):
    with pytest.raises(ValueError):
        faults.apply(x, "visible_rain", 2)
