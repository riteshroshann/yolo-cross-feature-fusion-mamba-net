"""Sensor-failure benchmark: physically motivated faults on (b, 4, H, W) visible+thermal batches in [0, 1]."""
import torch
import torch.nn.functional as F

SEVERITIES = (1, 2, 3)


def _blur(x, k):
    w = torch.ones(x.shape[1], 1, k, k, device=x.device, dtype=x.dtype) / (k * k)
    return F.conv2d(F.pad(x, (k // 2,) * 4, mode="replicate"), w, groups=x.shape[1])


def _vis_dark(v, s, g):
    return (v * (0.35, 0.15, 0.05)[s - 1] + (0.01, 0.02, 0.03)[s - 1] * torch.randn(v.shape, generator=g, device=v.device)).clamp(0, 1)


def _vis_glare(v, s, g):
    b, _, h, w = v.shape
    ys = torch.linspace(-1, 1, h, device=v.device)[:, None]
    xs = torch.linspace(-1, 1, w, device=v.device)[None]
    c = torch.rand(b, 2, generator=g, device=v.device) * 1.2 - 0.6
    r = (0.35, 0.55, 0.8)[s - 1]
    d2 = (ys[None] - c[:, :1, None]) ** 2 + (xs[None] - c[:, 1:, None]) ** 2
    bloom = torch.exp(-d2 / (2 * r * r))[:, None]
    return (v + bloom * (0.6, 0.9, 1.2)[s - 1]).clamp(0, 1)


def _vis_fog(v, s, g):
    t = (0.65, 0.4, 0.2)[s - 1]
    return v * t + 0.75 * (1 - t)


def _vis_noise(v, s, g):
    return (v + (0.06, 0.12, 0.2)[s - 1] * torch.randn(v.shape, generator=g, device=v.device)).clamp(0, 1)


def _vis_blur(v, s, g):
    return _blur(v, (5, 9, 15)[s - 1])


def _drop(x, s, g):
    return torch.zeros_like(x)


def _th_shift(t, s, g):
    k = (4, 8, 16)[s - 1]
    return torch.roll(t, shifts=(k, k), dims=(2, 3))


def _th_noise(t, s, g):
    b, _, h, w = t.shape
    cols = (0.04, 0.08, 0.14)[s - 1] * torch.randn(b, 1, 1, w, generator=g, device=t.device)
    return (t + cols + (0.03, 0.06, 0.1)[s - 1] * torch.randn(t.shape, generator=g, device=t.device)).clamp(0, 1)


def _th_crossover(t, s, g):
    a = (0.5, 0.25, 0.1)[s - 1]
    m = t.mean(dim=(2, 3), keepdim=True)
    return (m + (t - m) * a).clamp(0, 1)


def _th_dead(t, s, g):
    p = (0.02, 0.06, 0.15)[s - 1]
    mask = torch.rand(t.shape, generator=g, device=t.device) < p
    return torch.where(mask, torch.zeros_like(t), t)


VISIBLE = {"dark": _vis_dark, "glare": _vis_glare, "fog": _vis_fog, "noise": _vis_noise, "blur": _vis_blur,
           "drop": _drop}
THERMAL = {"drop": _drop, "shift": _th_shift, "noise": _th_noise, "crossover": _th_crossover, "dead": _th_dead}
FAULTS = tuple(f"visible_{k}" for k in VISIBLE) + tuple(f"thermal_{k}" for k in THERMAL)


def apply(x: torch.Tensor, fault: str, severity: int = 2, seed: int = 0) -> torch.Tensor:
    """Apply one fault at severity 1-3; 'clean' returns x unchanged."""
    if fault == "clean":
        return x
    sensor, kind = fault.split("_", 1)
    table = VISIBLE if sensor == "visible" else THERMAL if sensor == "thermal" else None
    if table is None or kind not in table or severity not in SEVERITIES:
        raise ValueError(f"unknown fault {fault!r} at severity {severity}; faults: {FAULTS}")
    g = torch.Generator(device=x.device).manual_seed(seed)
    x = x.clone()
    sl = slice(0, 3) if sensor == "visible" else slice(3, None)
    x[:, sl] = table[kind](x[:, sl], severity, g)
    return x


def flags_for(fault: str):
    """Sensor-health flags a monitor would raise: only a dropped stream is detectable for sure."""
    return {"visible_drop": [[0.0, 1.0]], "thermal_drop": [[1.0, 0.0]]}.get(fault)
