"""
Glue for running the pilot as a set of notebooks, on Kaggle or locally.

Kaggle gives every notebook a fresh, empty /kaggle/working and mounts inputs
read-only under /kaggle/input/<name>/. So each notebook has to be able to:

  * find the raw datasets wherever they were mounted        locate_raw()
  * convert them once, or reuse a conversion already done   prepare()
  * find checkpoints produced by earlier notebooks          find_checkpoints()
  * read the run plan, so every notebook trains the same    load_plan(), run_spec()
    configuration with the same settings

All run settings live in configs/pilot.yaml. Notebooks never hard-code an
epoch count or a batch size; change the YAML and every notebook follows.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import yaml

from . import data as cdata

INPUT_ROOTS = [Path("/kaggle/input"), Path("/content/drive/MyDrive")]


# --------------------------------------------------------------------------- #
# the run plan
# --------------------------------------------------------------------------- #
def load_plan(project: Path) -> dict:
    return yaml.safe_load((Path(project) / "configs" / "pilot.yaml").read_text())


def run_spec(plan: dict, name: str) -> dict:
    """Defaults merged with one run's settings."""
    run = next((r for r in plan["runs"] if r["name"] == name), None)
    if run is None:
        raise KeyError(f"no run named {name!r} in configs/pilot.yaml")
    return {**plan["defaults"], **run}


def runs_for(plan: dict, notebook: str) -> list[dict]:
    return [run_spec(plan, r["name"]) for r in plan["runs"] if str(r.get("notebook")) == str(notebook)]


# --------------------------------------------------------------------------- #
# finding raw data
# --------------------------------------------------------------------------- #
def copy_writable(src: Path, dst: Path) -> Path:
    """Copy a folder out of a read-only input. Plain shutil.copytree keeps the read-only modes, and then
    the YAMLs cannot be rewritten, Ultralytics cannot write its label caches and a run cannot resume."""
    shutil.copytree(src, dst, copy_function=shutil.copyfile, dirs_exist_ok=True)
    for p in [dst, *dst.rglob("*")]:
        p.chmod(p.stat().st_mode | 0o200)
    return dst


def _roots(env) -> list[Path]:
    return [p for p in [env.raw, *INPUT_ROOTS] if p.exists()]


def _looks_like_llvip(p: Path) -> bool:
    return (p / "Annotations").is_dir() and (p / "visible").is_dir() and (p / "infrared").is_dir()


def _looks_like_m3fd(p: Path) -> bool:
    if not p.is_dir() or _looks_like_llvip(p):  # LLVIP's visible/infrared/Annotations would pass the test below
        return False
    names = {c.name.lower() for c in p.iterdir() if c.is_dir()}
    return bool(names & {"vis", "visible"}) and bool(names & {"ir", "infrared"}) and \
        bool(names & {"annotation", "annotations", "labels"})


CHECKS = {"llvip": _looks_like_llvip, "m3fd": _looks_like_m3fd}


def locate_raw(name: str, env, max_depth: int = 5) -> Path | None:
    """Search the raw folder and Kaggle/Colab inputs for a dataset's raw layout."""
    check = CHECKS[name]
    for root in _roots(env):
        stack = [(root, 0)]
        while stack:
            p, depth = stack.pop()
            try:
                if check(p):
                    return p
                if depth < max_depth:
                    stack.extend((c, depth + 1) for c in p.iterdir() if c.is_dir())
            except (PermissionError, OSError):
                continue
    return None


def find_converted(name: str, env) -> Path | None:
    """A conversion we already did: in env.data, or attached as an input (an earlier notebook's output)."""
    local = env.data / name
    if (local / "dataset_card.json").exists():
        return local
    for root in INPUT_ROOTS:
        if root.exists():
            for card in root.glob(f"**/{name}/dataset_card.json"):
                return card.parent
    return None


def prepare(name: str, env, plan: dict | None = None, force: bool = False) -> Path:
    """Return a converted dataset folder, converting from raw only if needed.

    A conversion found under /kaggle/input is copied to /kaggle/working first,
    because data YAMLs carry absolute paths and inputs are read-only.
    """
    opts = ((plan or {}).get("datasets") or {}).get(name, {})
    out = env.data / name
    if not force:
        found = find_converted(name, env)
        if found is not None:
            if found != out:
                copy_writable(found, out)
            cdata.write_yamls(out, json.loads((out / "dataset_card.json").read_text())["names"])  # refresh paths
            return out
    raw = locate_raw(name, env)
    if raw is None:
        raise FileNotFoundError(
            f"raw {name} not found. Attach it as a Kaggle input (or put it under {env.raw}); "
            f"see notebooks/phase0_environment/01_data_acquisition.ipynb")
    convert = {"llvip": cdata.convert_llvip, "m3fd": cdata.convert_m3fd}[name]
    card = convert(raw, out, **opts)
    print(json.dumps(card, indent=2))
    return out


# --------------------------------------------------------------------------- #
# finding checkpoints and results from earlier notebooks
# --------------------------------------------------------------------------- #
def find_checkpoints(env) -> dict[str, Path]:
    """{run name: best.pt (or last.pt)} from this session and from attached notebook outputs."""
    found = {}
    for root in [env.runs, *[r for r in INPUT_ROOTS if r.exists()]]:
        for w in sorted(root.glob("**/weights/best.pt")) + sorted(root.glob("**/weights/last.pt")):
            found.setdefault(w.parent.parent.name, w)
    return found


def find_prior_run(name: str) -> Path | None:
    """The folder of run `name` in an attached input (an earlier version of this notebook), if any.
    Prefers the copy with the most finished epochs."""
    best, best_rows = None, -1
    for root in [r for r in INPUT_ROOTS if r.exists()]:
        for w in root.glob(f"**/{name}/weights/last.pt"):
            run = w.parent.parent
            try:
                rows = len((run / "results.csv").read_text().strip().splitlines())
            except OSError:
                rows = 0
            if rows > best_rows:
                best, best_rows = run, rows
    return best


def find_metrics(env, kind: str = "eval") -> dict[str, dict]:
    """{name: metrics} for every metrics.json under runs/<kind>/ here and in attached inputs."""
    out = {}
    for root in [env.runs, *[r for r in INPUT_ROOTS if r.exists()]]:
        for f in sorted(root.glob(f"**/{kind}/*/metrics.json")):
            out.setdefault(f.parent.name, json.loads(f.read_text()))
    return out


def train_from_spec(spec: dict, env, device=None, evaluate_after: bool = True, **override) -> dict:
    """Train one run from configs/pilot.yaml, then evaluate it on the full validation set.

    Returns {'run': name, 'weights': path, 'metrics': {...}}. Skips training if
    a finished checkpoint for this run already exists (re-running a notebook
    after a Kaggle session ends does not start from zero).
    """
    from .env import save_versions
    from .train import evaluate, train_run

    spec = {**spec, **override}
    name = spec["name"]
    model = spec["model"]
    model_path = str(env.project / model) if model.startswith("configs/") else model
    data_yaml = env.data / spec["data"]
    if not data_yaml.exists():
        prepare(spec["data"].split("/")[0], env)
    device = device if device is not None else env.device

    run_dir = env.runs / name
    if not run_dir.exists():
        prior = find_prior_run(name)
        if prior is not None:  # a new Kaggle session, with an earlier version's output attached
            print(f"[{name}] continuing from an attached output: {prior}")
            copy_writable(prior, run_dir)
    done = run_dir / "weights" / "best.pt"
    if done.exists() and (run_dir / "results.csv").exists() and _finished(run_dir, spec["epochs"]):
        print(f"[{name}] already trained, reusing {done}")
        best = done
    else:
        keys = ("epochs", "imgsz", "batch", "seed", "workers", "close_mosaic", "patience", "amp", "cache")
        kw = {k: spec[k] for k in keys if k in spec}
        resume_from = run_dir / "weights" / "last.pt"
        if resume_from.exists():  # a Kaggle session ended mid-run: pick up where it stopped
            print(f"[{name}] resuming from {resume_from}")
            kw = {"resume": str(resume_from)}
        best = train_run(name, model_path, str(data_yaml), pretrained=weights_file(env), device=device,
                         project=str(env.runs), **kw)
        save_versions(run_dir / "versions.json")
        (run_dir / "spec.json").write_text(json.dumps(spec, indent=2))

    metrics = {}
    if evaluate_after:
        metrics = evaluate(best, data_yaml, imgsz=spec["imgsz"], batch=max(8, spec["batch"] // 2),
                           device=str(device).split(",")[0], project=str(env.runs / "eval"), name=name)
        print(f"[{name}] AP50-95 {metrics.get('mAP50-95(B)', float('nan')):.4f}  "
              f"AP50 {metrics.get('mAP50(B)', float('nan')):.4f}  AP_s {metrics.get('AP_s(B)', float('nan')):.4f}")
    return {"run": name, "weights": str(best), "metrics": metrics}


def _finished(run_dir: Path, epochs: int) -> bool:
    """True if results.csv has all epochs (Ultralytics writes one row per epoch)."""
    try:
        rows = (run_dir / "results.csv").read_text().strip().splitlines()
        return len(rows) - 1 >= epochs
    except OSError:
        return False


def weights_file(env, name: str = "yolo26n.pt") -> str:
    """Local copy of the COCO weights if present, else the bare name (Ultralytics downloads it)."""
    p = env.weights / name
    return str(p) if p.exists() else name


def download_gdrive(url_or_id: str, dest: Path) -> Path:
    """Download a public Google Drive file (e.g. an official dataset link) and unzip it."""
    import subprocess
    import sys

    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "gdown"], check=True)
    import gdown

    dest.mkdir(parents=True, exist_ok=True)
    out = gdown.download(url_or_id if url_or_id.startswith("http") else None,
                         id=None if url_or_id.startswith("http") else url_or_id,
                         output=str(dest) + "/", quiet=False, fuzzy=True)
    out = Path(out)
    if out.suffix.lower() in {".zip", ".tar", ".gz", ".tgz"}:
        shutil.unpack_archive(str(out), str(dest))
    return dest
