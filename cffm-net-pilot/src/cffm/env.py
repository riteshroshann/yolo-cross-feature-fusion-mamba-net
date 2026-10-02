"""
Where am I running, and where do things live?

Every notebook starts with `env = cffm.env.setup()`. It works out whether we
are on Kaggle, Colab or a local machine, picks the folders for data, runs and
weights, and reports the GPU so that batch sizes can be chosen sensibly.
Nothing else in the code base hard-codes a path.
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
    platform: str                      # 'kaggle' | 'colab' | 'local'
    project: Path                      # the cffm-net-pilot folder
    data: Path                         # converted datasets
    raw: Path                          # downloaded archives and raw folders
    runs: Path                         # training and evaluation outputs
    weights: Path                      # pretrained checkpoints (yolo26n.pt)
    device: str                        # '0', '0,1' or 'cpu'
    gpus: list = field(default_factory=list)   # [(name, total GiB), ...]
    scan_impl: str = "torch"           # 'cuda' if the fused mamba_ssm kernel is usable

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
    if plat == "kaggle":            # /kaggle/input is read-only; everything we write goes to /kaggle/working
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
    os.chdir(project)               # relative paths in configs resolve from the project folder
    if verbose:
        print(env.summary())
    return env


def suggested_batch(env: Env, dual: bool, imgsz: int = 640) -> int:
    """A safe starting batch size for YOLO26-n at 640 px. Ultralytics' batch=-1 can refine it.

    Rough rule from GPU memory: a single-stream YOLO26-n at 640 needs about
    0.25 GB per image in training with AMP; the dual-stream CFFM-Net with the
    stride-4 head about 0.6 GB per image with the PyTorch scan.
    """
    if not env.gpus:
        return 2
    mem = min(m for _, m in env.gpus) - 1.0                 # leave 1 GB for the CUDA context
    per_img = (0.6 if dual else 0.25) * (imgsz / 640) ** 2
    b = max(2, int(mem / per_img))
    b = 2 ** int(b).bit_length() // 2 if b & (b - 1) else b  # round down to a power of two
    return min(b, 64) * max(1, len(env.gpus))               # Ultralytics splits the batch over GPUs


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
