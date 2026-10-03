"""Platform detection (Kaggle, Colab or local), data/run paths and the GPU report.

Every notebook starts with `env = cffm.env.setup()`; nothing else hard-codes a path.
"""
from __future__ import annotations

import json
import os
import platform
import sys
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Env:
    platform: str
    project: Path
    data: Path
    raw: Path
    runs: Path
    weights: Path
    device: str
    gpus: list = field(default_factory=list)
    scan_impl: str = "torch"

    def summary(self) -> str:
        g = ", ".join(f"{n} ({m:.0f} GB)" for n, m in self.gpus) or "no GPU"
        return (f"platform={self.platform}  device={self.device}  gpus=[{g}]  scan={self.scan_impl}\n"
                f"project={self.project}\ndata={self.data}\nruns={self.runs}")


def _platform() -> str:
    if os.environ.get("KAGGLE_KERNEL_RUN_TYPE") or Path("/kaggle/working").exists():
        return "kaggle"
    if "google.colab" in sys.modules or os.environ.get("COLAB_RELEASE_TAG"):
        return "colab"
    return "local"


def find_project() -> Path:
    """The folder that contains src/cffm, searched upwards from here and from the cwd."""
    for start in (Path(__file__).resolve(), Path.cwd().resolve()):
        for p in [start, *start.parents]:
            if (p / "src" / "cffm").is_dir():
                return p
    raise FileNotFoundError("cannot find the cffm-net-pilot folder (looked for src/cffm)")


def setup(verbose: bool = True) -> Env:
    import torch

    plat = _platform()
    project = find_project()
    if plat == "kaggle":
        root = Path("/kaggle/working")
        data, raw, runs = root / "data", root / "raw", root / "runs"
    else:
        data, raw, runs = project / "data" / "processed", project / "data" / "raw", project / "runs"
    weights = project / "weights"
    for p in (data, raw, runs, weights):
        p.mkdir(parents=True, exist_ok=True)

    gpus = []
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            prop = torch.cuda.get_device_properties(i)
            gpus.append((prop.name, prop.total_memory / 2 ** 30))
    device = ",".join(str(i) for i in range(len(gpus))) if gpus else "cpu"

    from . import scan
    scan_impl = "cuda" if (scan.HAS_MAMBA_SSM and gpus) else "torch"
    env = Env(plat, project, data, raw, runs, weights, device, gpus, scan_impl)
    os.chdir(project)
    if verbose:
        print(env.summary())
    return env


def suggested_batch(env: Env, dual: bool, imgsz: int = 640) -> int:
    """A safe starting batch size: about 0.25 GB per image single-stream, 0.6 GB dual, at 640 px with AMP."""
    if not env.gpus:
        return 2
    mem = min(m for _, m in env.gpus) - 1.0
    per_img = (0.6 if dual else 0.25) * (imgsz / 640) ** 2
    b = max(2, int(mem / per_img))
    b = 2 ** int(b).bit_length() // 2 if b & (b - 1) else b
    return min(b, 64) * max(1, len(env.gpus))


def versions() -> dict:
    """Library versions, recorded with every run."""
    import torch
    import ultralytics

    from . import scan
    v = {"python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda,
         "ultralytics": ultralytics.__version__, "mamba_ssm": scan.HAS_MAMBA_SSM, "os": platform.platform()}
    return v


def save_versions(path: Path):
    Path(path).write_text(json.dumps(versions(), indent=2))
