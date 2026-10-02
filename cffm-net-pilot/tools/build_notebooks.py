"""
Generate every pilot notebook from one script.

Why generate notebooks instead of editing them by hand? Because eleven
notebooks that share a setup cell drift apart the moment one is edited. Here
the shared parts are written once, the notebooks are rebuilt in a second, and
a diff of this file is a readable diff of the whole workflow.

    python tools/build_notebooks.py

Notebooks are written without outputs; they are meant to be run on Kaggle.
"""
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]


def md(text):
    return nbf.v4.new_markdown_cell(text.strip("\n"))


def code(text):
    return nbf.v4.new_code_cell(text.strip("\n"))


def write(path, cells):
    nb = nbf.v4.new_notebook()
    nb.cells = cells
    nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}}
    out = ROOT / path
    out.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, out)
    print("wrote", path)


# --------------------------------------------------------------------------- #
# the cell every notebook starts with
# --------------------------------------------------------------------------- #
SETUP = code('''
# --------------------------------------------------------------------------------------------
# Setup: the same cell opens every notebook. It finds the code, installs it, and checks the GPU.
#
# On Kaggle the uploaded cffm-net-pilot.zip appears, already unzipped, somewhere under
# /kaggle/input. Inputs are read-only, so we copy the project to /kaggle/working and install
# it from there ("pip install -e" makes `import cffm` work in this kernel, in DataLoader
# workers, and in the extra processes Ultralytics starts when it trains on two GPUs).
# Locally, notebooks live in cffm-net-pilot/notebooks/<phase>/; the project is the first folder up
# that contains src/cffm.
# --------------------------------------------------------------------------------------------
import glob, shutil, subprocess, sys
from pathlib import Path

ON_KAGGLE = Path("/kaggle/working").exists()
if ON_KAGGLE:
    # up to five levels deep: /kaggle/input/<slug>/, /kaggle/input/<slug>/cffm-net-pilot/,
    # /kaggle/input/datasets/<owner>/<slug>/..., whichever layout this Kaggle version uses
    hits = [Path(p).parent for k in range(1, 6) for p in glob.glob("/kaggle/input" + "/*" * k + "/pyproject.toml")
            if (Path(p).parent / "src" / "cffm").is_dir()]
    assert hits, "Add the 'cffm-net-pilot' dataset (the uploaded zip) as an input to this notebook."
    PROJECT = Path("/kaggle/working/cffm-net-pilot")
    if not PROJECT.exists():
        shutil.copytree(hits[0], PROJECT, copy_function=shutil.copyfile)
        for p in [PROJECT, *PROJECT.rglob("*")]:     # the input was read-only; our copy must be writable
            p.chmod(p.stat().st_mode | 0o200)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ultralytics==8.4.171",
                    "faster-coco-eval>=1.6.7", "cloudpickle"], check=True)   # cloudpickle: two-GPU launcher
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "-e", str(PROJECT)], check=True)
else:
    PROJECT = next(p for p in [Path.cwd().resolve(), *Path.cwd().resolve().parents] if (p / "src" / "cffm").is_dir())
sys.path.insert(0, str(PROJECT / "src"))

import cffm
from cffm import env as cenv, pipeline
print("cffm", cffm.__version__, "imported from", Path(cffm.__file__).parent)   # must be inside PROJECT
env = cenv.setup()                       # prints platform, GPUs and paths; moves into the project folder
plan = pipeline.load_plan(env.project)   # configs/pilot.yaml: every run and its settings
''')


def header(title, purpose, inputs, runtime, outputs):
    return md(f"""
# {title}

{purpose}

| | |
|---|---|
| **Kaggle settings** | Accelerator **GPU T4 x2**, Internet **on** |
| **Inputs to attach** | {inputs} |
| **Expected runtime** | {runtime} |
| **Produces** | {outputs} |
""")


def figure(*names, width=820):
    # A code cell that shows methodology diagrams from docs/figures (built by docs/figures/build_figures.py).
    return code(f"""
from IPython.display import Image, display
for name in {names!r}:
    f = env.project / "docs" / "figures" / f"{{name}}.png"
    if f.exists():
        display(Image(filename=str(f), width={width}))
    else:
        print("missing", f)
""")


TRAIN_RUNS = '''
import pandas as pd
cols = ["name", "model", "data", "epochs", "imgsz", "batch"]
pd.DataFrame(specs)[cols]
'''

TRAIN_LOOP = '''
# Train every run of this notebook, then evaluate each on the FULL validation set.
# If the Kaggle session dies halfway, just run the notebook again: finished runs are reused
# and an unfinished run resumes from its last checkpoint (see pipeline.train_from_spec).
results = [pipeline.train_from_spec(s, env) for s in specs]
'''

CURVES = '''
# Learning curves straight from Ultralytics' results.csv. Two things to look for:
#   - validation mAP still climbing at the last epoch -> the pilot schedule is short (expected)
#   - training loss going down while validation mAP goes down too -> overfitting (not expected in 30 epochs)
import matplotlib.pyplot as plt
import pandas as pd

fig, axes = plt.subplots(1, 3, figsize=(16, 3.8))
for s in specs:
    f = env.runs / s["name"] / "results.csv"
    if not f.exists():
        continue
    df = pd.read_csv(f)
    df.columns = [c.strip() for c in df.columns]
    axes[0].plot(df["epoch"], df["metrics/mAP50-95(B)"], label=s["name"])
    axes[1].plot(df["epoch"], df["metrics/mAP50(B)"], label=s["name"])
    loss_cols = [c for c in df.columns if c.startswith("train/") and c.endswith("loss")]
    axes[2].plot(df["epoch"], df[loss_cols].sum(axis=1), label=s["name"])
for a, t in zip(axes, ["val mAP50-95 (quarter val set)", "val mAP50", "train loss (sum)"]):
    a.set_title(t); a.set_xlabel("epoch"); a.grid(alpha=0.3)
axes[0].legend(fontsize=8)
plt.tight_layout(); plt.show()
'''

SUMMARY = '''
# The numbers that matter, from the full validation set. AP is COCO AP50-95.
# AP_vt / AP_t / AP_s are the AI-TOD size bands (<8, 8-16, 16-32 px); mAP_small is COCO's <32 px.
import pandas as pd
keys = ["mAP50-95(B)", "mAP50(B)", "mAP_small(B)", "AP_vt(B)", "AP_t(B)", "AP_s(B)", "AP_m(B)", "AP_l(B)"]
rows = [{"run": r["run"], **{k.replace("(B)", ""): r["metrics"].get(k) for k in keys}} for r in results]
pd.DataFrame(rows).set_index("run").round(4)
'''

SAVE_NOTE = md('''
### Keep the results

On Kaggle, click **Save Version → Save & Run All (Commit)**. The notebook then reruns top to bottom in the
background, and its `runs/` folder becomes an output that notebooks **09** and **15** can attach as an input
(*Add Input → Your Work → this notebook*). Without a saved version, the checkpoints vanish when the session ends.
''')


# --------------------------------------------------------------------------- #
# Phase 0
# --------------------------------------------------------------------------- #
def nb00():
    write("notebooks/phase0_environment/00_environment_setup.ipynb", [
        header("00 · Environment setup and checks",
               "Phase 0 of the CFFM-Net pilot. Before spending a single GPU-hour on training we prove, on this "
               "exact machine, that every moving part works: the libraries, the GPUs, the selective scan, the "
               "model, the paired data loader, two-GPU training, evaluation and the degradation probe. If "
               "something is broken this is the cheapest place to find out: everything downstream would have "
               "failed too.",
               "`cffm-net-pilot` (the uploaded zip)", "about 10 to 15 minutes",
               "a go / no-go answer, measured scan speed and memory, and `weights/yolo26n.pt`"),
        SETUP,
        md("## 0. The pilot at a glance\n\nEleven notebooks in two phases. Solid arrows carry data and checkpoints; "
           "dashed arrows carry metrics. The four training notebooks (04, 05, 07, 08) are independent Kaggle "
           "sessions. Notebooks 09 and 15 read their saved outputs, and 15 decides the five pre-registered hypotheses."),
        figure("fig_workflow", width=900),
        md("## 1. What are we running on?"),
        code('''
import json, torch
print(json.dumps(cenv.versions(), indent=2))
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f"GPU {i}: {p.name}, {p.total_memory / 2**30:.1f} GB, compute capability {p.major}.{p.minor}")
# A T4 is compute capability 7.5. It has no bfloat16, so mixed precision uses float16, which is why
# cffm/scan.py always runs the recurrence itself in float32 (half precision loses long products of
# numbers below 1 very quickly).
assert torch.cuda.is_available(), "No GPU: set Accelerator to 'GPU T4 x2' in the notebook settings."
'''),
        md("## 2. COCO-pretrained YOLO26-n weights\n\nBoth backbones of every dual-stream model start from "
           "these weights, and the single-sensor baselines fine-tune them. We put one copy in `weights/` so "
           "every run uses the identical file."),
        code('''
import torch
w = env.weights / "yolo26n.pt"
if not w.exists():
    torch.hub.download_url_to_file("https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt", str(w))
print(w, f"{w.stat().st_size / 1e6:.1f} MB")
'''),
        md('''
## 3. (Optional) the fused Mamba kernel

The pilot does **not** need it. `cffm/scan.py` contains a pure-PyTorch selective scan that runs on any GPU
and is tested against a naive loop. The fused `mamba_ssm` kernel is faster, but it compiles from source
(20+ minutes) and may not support this PyTorch build or the T4. Leave the flag `False` unless you want to
try; if it installs, restart the kernel and the code picks it up automatically (`scan_impl: auto`).
'''),
        code('''
TRY_FUSED_SCAN = False
if TRY_FUSED_SCAN:
    r = subprocess.run([sys.executable, "-m", "pip", "install", "--no-build-isolation", "causal-conv1d", "mamba-ssm"],
                       capture_output=True, text=True)
    print(r.stdout[-3000:], r.stderr[-3000:])
    print("Now restart the kernel and run the notebook again from the top.")
from cffm import scan
print("fused kernel available:", scan.HAS_MAMBA_SSM)
'''),
        md('''
## 4. Unit tests

Sixty-two small tests, each checking one claim the method depends on. Two need the optional fused kernel and
four are an opt-in smoke run (section 7 below does that run itself), so expect 56 passed and 6 skipped.

* the chunked scan equals the textbook recurrence, values *and* gradients;
* a token with step 0 cannot write into the state (the property reliability gating relies on);
* the scan's token order really interleaves visible and thermal, and every direction maps back exactly;
* a fresh CMFM block is exactly a reliability-weighted average (its output projection starts at zero);
* every model config builds and runs, and COCO weights fill **both** backbones;
* paired loading returns 4 registered channels;
* the Phase 2 Temporal State Memory holds, resets and warps as specified.
'''),
        code('''
r = subprocess.run([sys.executable, "-m", "pytest", "-q", "tests"], cwd=env.project, capture_output=True, text=True)
print(r.stdout[-4000:])
assert r.returncode == 0, r.stderr[-4000:]
'''),
        md('''
## 5. How fast is the scan here?

The biggest scan in CFFM-Net sits at stride 8. A 640 x 640 training crop gives an 80 x 80 map; interleaving
visible and thermal doubles it to **12 800 tokens**; four directions of 64 channels make **256 scan
channels**; and Kaggle trains **8 images per GPU**. That is the size we time below, forward and backward.
'''),
        code('''
import time, torch
from cffm.scan import selective_scan

def time_scan(b=8, d=256, L=12800, n=8, groups=4, impl="torch", reps=3):
    dev = "cuda:0"
    u = torch.randn(b, d, L, device=dev, requires_grad=True)
    delta = torch.nn.functional.softplus(torch.randn(b, d, L, device=dev))
    A = -torch.exp(torch.randn(d, n, device=dev))
    B, C = torch.randn(b, groups, n, L, device=dev), torch.randn(b, groups, n, L, device=dev)
    D = torch.ones(d, device=dev)
    torch.cuda.reset_peak_memory_stats(); times = []
    for _ in range(reps + 1):
        torch.cuda.synchronize(); t = time.time()
        y = selective_scan(u, delta, A, B, C, D, impl=impl, **({"use_checkpoint": True} if impl == "torch" else {}))
        y.sum().backward(); torch.cuda.synchronize(); times.append(time.time() - t)
    return 1000 * sorted(times[1:])[len(times[1:]) // 2], torch.cuda.max_memory_allocated() / 2**30

ms, gb = time_scan()
print(f"PyTorch scan, stride-8 size, batch 8: {ms:.0f} ms forward+backward, peak {gb:.2f} GB")
if scan.HAS_MAMBA_SSM:
    ms, gb = time_scan(impl="cuda")
    print(f"fused kernel, same size:            {ms:.0f} ms forward+backward, peak {gb:.2f} GB")
'''),
        md('''
## 6. Do the planned batch sizes fit?

One forward and backward pass per model at its planned batch per GPU (from `configs/pilot.yaml`), with mixed
precision and random inputs. This is not training: it only measures memory and step time, so we know before
the real runs whether a batch size has to come down. The last column projects one 30-epoch run on LLVIP's
~4 000 training pairs.
'''),
        code('''
import time, torch, pandas as pd
from ultralytics.nn.tasks import DetectionModel
from cffm.model import DualStreamDetectionModel

n_gpu = max(1, torch.cuda.device_count())
def tensors(o):
    """Every tensor inside nested dicts / lists / tuples (the head's training output is nested)."""
    if torch.is_tensor(o): return [o]
    if isinstance(o, dict): o = list(o.values())
    return [t for x in o for t in tensors(x)] if isinstance(o, (list, tuple)) else []

def step(model, b, ch, hw=(640, 640), reps=3):
    m = model.to("cuda:0").train(); opt = torch.optim.SGD(m.parameters(), lr=1e-4)
    x = torch.rand(b, ch, *hw, device="cuda:0")
    torch.cuda.reset_peak_memory_stats(); times = []
    for _ in range(reps + 1):
        torch.cuda.synchronize(); t = time.time()
        with torch.autocast("cuda", dtype=torch.float16):
            loss = sum(t.float().mean() for t in tensors(m(x)) if t.requires_grad)   # a stand-in loss
        loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize(); times.append(time.time() - t)
    return sorted(times[1:])[len(times[1:]) // 2], torch.cuda.max_memory_allocated() / 2**30

rows, seen = [], set()
for s in (pipeline.run_spec(plan, r["name"]) for r in plan["runs"]):
    if s["model"] in seen:
        continue
    seen.add(s["model"])
    dual = s["model"].startswith("configs/")
    model = DualStreamDetectionModel(str(env.project / s["model"]), nc=1, verbose=False) if dual \\
        else DetectionModel(s["model"], nc=1, verbose=False)
    per_gpu = s["batch"] // n_gpu
    t, gb = step(model, per_gpu, 4 if dual else 3)
    hours = 4000 / s["batch"] * t * s["epochs"] / 3600 * 1.15      # +15% for data loading and DDP sync
    rows.append({"model": Path(s["model"]).stem, "batch/GPU": per_gpu, "peak GB": round(gb, 2),
                 "step s": round(t, 3), "~hours for 30 ep on LLVIP": round(hours, 2)})
    del model; torch.cuda.empty_cache()
df = pd.DataFrame(rows); print(df.to_string(index=False))
limit = min(torch.cuda.get_device_properties(i).total_memory for i in range(n_gpu)) / 2**30
assert (df["peak GB"] < limit - 1.0).all(), "A batch is too large for this GPU: lower it in configs/pilot.yaml."
'''),
        md('''
## 7. The whole pipeline, end to end, on fake data and both GPUs

A tiny synthetic paired dataset (bright squares that are hot in thermal), one epoch per model, with the
real trainer, the two-GPU (DDP) launcher, the validator with size-binned COCO metrics, and the degradation
probe. The numbers mean nothing; the point is that no step crashes on this machine.
'''),
        code('''
from cffm.data import make_synthetic
from cffm.train import evaluate, probe, train_run

synth = make_synthetic(env.data / "synthetic", n_train=32, n_val=8, imgsz=320)
smoke = {}
for name, model, data in [("smoke_cffm", str(env.project / "configs/models/cffm-net-n.yaml"), "data_paired.yaml"),
                          ("smoke_concat", str(env.project / "configs/models/two-stream-concat-n.yaml"), "data_paired.yaml"),
                          ("smoke_visible", "yolo26n.yaml", "data_visible.yaml")]:
    best = train_run(name, model, str(synth / data), pretrained=pipeline.weights_file(env), epochs=1, imgsz=320,
                     batch=8, device=env.device, project=str(env.runs / "smoke"), workers=2, plots=False)
    smoke[name] = evaluate(best, synth / data, imgsz=320, batch=8, device=0, project=str(env.runs / "smoke_eval"))
    print(name, "ok:", {k: round(v, 3) for k, v in smoke[name].items() if k in ("mAP50(B)", "AP_s(B)")})

best = env.runs / "smoke" / "smoke_cffm" / "weights" / "best.pt"
for kind, flagged in [("clean", False), ("thermal_drop", False), ("thermal_drop", True), ("thermal_shift_8", False)]:
    m = probe(best, synth / "data_paired.yaml", kind, flagged=flagged, imgsz=320, batch=8, device=0,
              project=str(env.runs / "smoke_probe"))
    print("probe", kind, "flagged" if flagged else "", "ok")
'''),
        md('''
## 8. Verdict

If every cell above ran without an error, the environment is ready and the code is verified on this
hardware. Clean up the smoke runs (they are not results), then continue with
**01 · Data acquisition**.
'''),
        code('''
shutil.rmtree(env.runs / "smoke", ignore_errors=True)
for d in ("smoke_eval", "smoke_probe"):
    shutil.rmtree(env.runs / d, ignore_errors=True)
shutil.rmtree(env.data / "synthetic", ignore_errors=True)
print("Phase 0 checks passed. Next: 01_data_acquisition.ipynb")
'''),
    ])


def nb01():
    write("notebooks/phase0_environment/01_data_acquisition.ipynb", [
        header("01 · Data acquisition",
               "The pilot uses the two paired visible-thermal benchmarks that are freely available: **LLVIP** "
               "(15 488 aligned pairs of pedestrians, mostly at night) and **M3FD** (4 200 aligned pairs, six "
               "classes, mixed scenes). This notebook gets them onto the machine and checks they are complete. "
               "Both are licensed for non-commercial research; do not make your Kaggle copies public.",
               "`cffm-net-pilot`, plus LLVIP and M3FD (see below)", "a few minutes (longer if downloading)",
               "verified raw datasets"),
        md('''
## How to get the data onto Kaggle

**Option A (recommended).** Download the official archives on your computer and upload each as a **private**
Kaggle dataset (*Datasets → New Dataset → upload the zip*; Kaggle unzips it). Then attach both here
(*Add Input → Your Work*).

* LLVIP: official page <https://bupt-ai-cz.github.io/LLVIP/> (links to the GitHub release). Use the
  annotations corrected in February 2023 (the current release has them).
* M3FD: from the TarDAL repository <https://github.com/JinyuanLiu-CV/TarDAL> (the *M3FD_Detection* part:
  `Vis/`, `Ir/`, `Annotation/`).

**Option B.** Attach an existing public Kaggle copy (search for "LLVIP" / "M3FD"). Check that it has the
official folder layout and the corrected LLVIP labels.

**Option C.** Paste a public Google Drive link or file ID below and download inside this notebook.

The code finds the datasets by their folder layout, wherever they are mounted, so names do not matter.
'''),
        SETUP,
        code('''
# Option C only: paste Google Drive links or file IDs of the official archives (leave empty otherwise).
LLVIP_GDRIVE = ""
M3FD_GDRIVE = ""
if LLVIP_GDRIVE:
    pipeline.download_gdrive(LLVIP_GDRIVE, env.raw / "llvip")
if M3FD_GDRIVE:
    pipeline.download_gdrive(M3FD_GDRIVE, env.raw / "m3fd")
'''),
        md("## Where are they?"),
        code('''
found = {name: pipeline.locate_raw(name, env) for name in ("llvip", "m3fd")}
for name, p in found.items():
    print(f"{name:6s} -> {p if p else 'NOT FOUND: attach it as an input (see the options above)'}")
'''),
        md("## Are they complete?\n\nCounts per folder, whether every image has its partner and its "
           "annotation, and one parsed annotation to see that the class names are what the converter expects."),
        code('''
from cffm.data import M3FD_NAMES, _parse_voc

def report_llvip(root):
    for split in ("train", "test"):
        vis = sorted((root / "visible" / split).glob("*.jpg"))
        ir = {p.name for p in (root / "infrared" / split).glob("*.jpg")}
        ann = {p.stem for p in (root / "Annotations").glob("*.xml")}
        paired = [v for v in vis if v.name in ir and v.stem in ann]
        print(f"LLVIP {split:5s}: {len(vis)} visible, {len(ir)} infrared, {len(paired)} complete pairs")
    x = next((root / "Annotations").glob("*.xml"))
    print("sample annotation", x.name, _parse_voc(x, ["person"])[:3])

def report_m3fd(root):
    from cffm.data import _find_dir
    vis, ir, ann = _find_dir(root, ["Vis", "visible"]), _find_dir(root, ["Ir", "infrared"]), \\
        _find_dir(root, ["Annotation", "Annotations", "Labels"])
    nv, ni, na = (len(list(d.iterdir())) for d in (vis, ir, ann))
    print(f"M3FD: {nv} visible, {ni} infrared, {na} annotations")
    x = next(ann.glob("*.xml"))
    print("sample annotation", x.name, _parse_voc(x, M3FD_NAMES)[:3])

if found["llvip"]: report_llvip(found["llvip"])
if found["m3fd"]: report_m3fd(found["m3fd"])
'''),
        md("## One pair from each\n\nIf the two images below do not show the same scene, the pairing is "
           "wrong and every result downstream would be meaningless."),
        code('''
import cv2, matplotlib.pyplot as plt
from cffm.data import _find_dir

def show_raw_pair(vis_path, ir_path, title):
    v, i = cv2.imread(str(vis_path))[..., ::-1], cv2.imread(str(ir_path), cv2.IMREAD_GRAYSCALE)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    ax[0].imshow(v); ax[0].set_title(f"{title}: visible"); ax[1].imshow(i, cmap="gray"); ax[1].set_title("thermal")
    for a in ax: a.axis("off")
    plt.show()

if found["llvip"]:
    v = sorted((found["llvip"] / "visible" / "test").glob("*.jpg"))[100]
    show_raw_pair(v, found["llvip"] / "infrared" / "test" / v.name, "LLVIP")
if found["m3fd"]:
    vd, idr = _find_dir(found["m3fd"], ["Vis", "visible"]), _find_dir(found["m3fd"], ["Ir", "infrared"])
    v = sorted(vd.iterdir())[50]
    show_raw_pair(v, next(idr.glob(v.stem + ".*")), "M3FD")
'''),
    ])


def nb02():
    write("notebooks/phase0_environment/02_data_conversion.ipynb", [
        header("02 · Data conversion",
               "Turns the raw datasets into one paired layout that Ultralytics can read: two folders of images "
               "with identical file names (`images/visible/...`, `images/infrared/...`), YOLO labels, and three "
               "data YAMLs per dataset (paired 4-channel, visible-only, thermal-only). Images are stored with "
               "their long side at 640 px, the resolution the models train at, which also makes loading fast. "
               "Every choice is written into a `dataset_card.json`.",
               "`cffm-net-pilot`, raw LLVIP and M3FD", "about 5 minutes",
               "`data/llvip/`, `data/m3fd/` (and, if you save a version, an attachable output)"),
        SETUP,
        md("## Convert\n\nThe options (which LLVIP frames to keep, the M3FD split seed) come from "
           "`configs/pilot.yaml`, so this notebook and every training notebook make the identical dataset."),
        code('''
cards = {}
for name in ("llvip", "m3fd"):
    out = pipeline.prepare(name, env, plan, force=False)
    cards[name] = __import__("json").loads((out / "dataset_card.json").read_text())
    print(name, "->", out)
'''),
        md("## What we made\n\nThe object-size histogram is the first look at how small these objects really "
           "are at the resolution the network sees."),
        code('''
import pandas as pd
rows = [{"dataset": n, "train": c["images"]["train"], "val (full)": c["images"]["val"], "objects": c["objects"],
         **c["object_size_bins"]} for n, c in cards.items()]
pd.DataFrame(rows).set_index("dataset")
'''),
        code('''
print((env.data / "llvip" / "data_paired.yaml").read_text())
'''),
        md("## Sanity checks\n\nEvery visible image has a thermal partner of the same size and a label file; "
           "every box lies inside the image."),
        code('''
import cv2, numpy as np
for name in ("llvip", "m3fd"):
    root = env.data / name
    for split in ("train", "val"):
        vis = sorted((root / "images" / "visible" / split).glob("*.jpg"))
        missing = [v for v in vis if not (root / "images" / "infrared" / split / v.name).exists()
                   or not (root / "labels" / "visible" / split / (v.stem + ".txt")).exists()]
        boxes = np.array([list(map(float, l.split()[1:])) for f in (root / "labels" / "visible" / split).glob("*.txt")
                          for l in f.read_text().splitlines() if l.strip()], dtype=float).reshape(-1, 4)
        inside = ((boxes[:, :2] - boxes[:, 2:] / 2 >= -1e-6) & (boxes[:, :2] + boxes[:, 2:] / 2 <= 1 + 1e-6)).all()
        # sizes: reading every image takes minutes, every 25th pair is plenty to catch a converter bug
        size_ok = all(cv2.imread(str(v)).shape[:2] == cv2.imread(str(root / "images" / "infrared" / split / v.name)).shape[:2]
                      for v in vis[::25])
        print(f"{name} {split}: {len(vis)} pairs, {len(missing)} incomplete, {len(boxes)} boxes, "
              f"all inside image: {bool(inside)}, pair sizes match: {size_ok}")
        assert vis and not missing and inside and size_ok
'''),
        md("## Look at a few converted pairs, with their labels\n\nGreen boxes on both images. If a box sits "
           "on a person in the visible image but beside them in the thermal one, the dataset is misregistered "
           "(notebook 03 measures this properly)."),
        code('''
import matplotlib.pyplot as plt
from cffm.data import read_pair
from cffm.viz import draw_pair

def show(name, k=3):
    root = env.data / name
    vis = sorted((root / "images" / "visible" / "val").glob("*.jpg"))
    vis = vis[:: max(1, len(vis) // k)][:k]                      # k pairs spread over the split
    names = cards[name]["names"]
    for v in vis:
        pair = read_pair(str(v)); h, w = pair.shape[:2]
        gt = []
        for line in (root / "labels" / "visible" / "val" / (v.stem + ".txt")).read_text().splitlines():
            c, x, y, bw, bh = map(float, line.split())
            gt.append([(x - bw / 2) * w, (y - bh / 2) * h, (x + bw / 2) * w, (y + bh / 2) * h, 1.0, c])
        plt.figure(figsize=(14, 4.5)); plt.imshow(draw_pair(pair, gt, names)); plt.axis("off"); plt.title(f"{name}: {v.name}")
        plt.show()

show("llvip"); show("m3fd")
'''),
        SAVE_NOTE,
    ])


# --------------------------------------------------------------------------- #
# Phase 1
# --------------------------------------------------------------------------- #
def nb03():
    write("notebooks/phase1_still_images/03_data_audit.ipynb", [
        header("03 · Data audit",
               "Three questions about the data, answered with numbers before any model sees it. How small are "
               "the objects? How dark are the scenes? And, most important for a fusion paper, how well are the "
               "two cameras really aligned? The alignment result also tells us how large the synthetic shifts in "
               "the degradation probe (notebook 09) should be.",
               "`cffm-net-pilot`, raw LLVIP and M3FD (or the output of 02)", "about 5 minutes",
               "`runs/audit/audit.json` and the figures below"),
        SETUP,
        code('''
for name in ("llvip", "m3fd"):
    pipeline.prepare(name, env, plan)
'''),
        md("## 1. Object sizes at training resolution\n\nSize is the square root of box area in pixels, after "
           "the long side was scaled to 640. The dashed lines are the AI-TOD bands (8, 16, 32 px) and COCO's "
           "small/medium boundary (96 px)."),
        code('''
import cv2, json, numpy as np, matplotlib.pyplot as plt
audit = {}
fig, axes = plt.subplots(1, 2, figsize=(14, 3.8))
for ax, name in zip(axes, ("llvip", "m3fd")):
    root = env.data / name
    card = json.loads((root / "dataset_card.json").read_text())
    H, W = cv2.imread(str(next((root / "images" / "visible" / "train").glob("*.jpg")))).shape[:2]   # stored size
    sizes, per_image, classes = [], [], np.zeros(len(card["names"]), int)
    for f in (root / "labels" / "visible" / "train").glob("*.txt"):
        lines = [l.split() for l in f.read_text().splitlines() if l.strip()]
        per_image.append(len(lines))
        for c, x, y, w, h in lines:
            classes[int(c)] += 1
            sizes.append(np.sqrt(float(w) * W * float(h) * H))       # labels are normalised to the image
    sizes = np.array(sizes)
    ax.hist(sizes, bins=np.logspace(0, 2.6, 60), color="#2F6DB5", alpha=0.8)
    ax.set_xscale("log"); ax.set_title(f"{name}: {len(sizes)} training objects"); ax.set_xlabel("object size (px)")
    for t in (8, 16, 32, 96): ax.axvline(t, ls="--", c="gray", lw=0.8)
    audit[name] = {"median_size_px": float(np.median(sizes)), "frac_below_16px": float((sizes < 16).mean()),
                   "frac_below_32px": float((sizes < 32).mean()), "objects_per_image": float(np.mean(per_image)),
                   "class_counts": dict(zip(card["names"], classes.tolist()))}
plt.tight_layout(); plt.show()
print(json.dumps(audit, indent=2))
'''),
        md("## 2. How dark are the scenes?\n\nMean brightness of the visible image. LLVIP is mostly night, which "
           "is exactly where a thermal camera should earn its place."),
        code('''
import cv2
fig, axes = plt.subplots(1, 2, figsize=(14, 3.2))
for ax, name in zip(axes, ("llvip", "m3fd")):
    files = sorted((env.data / name / "images" / "visible" / "val").glob("*.jpg"))[::5]
    b = np.array([cv2.imread(str(f), cv2.IMREAD_GRAYSCALE).mean() for f in files])
    ax.hist(b, bins=40, color="#C0622B", alpha=0.8); ax.set_title(f"{name}: mean visible brightness (0-255)")
    audit[name]["frac_dark_below_50"] = float((b < 50).mean())
plt.tight_layout(); plt.show()
'''),
        md('''
## 3. How well are the cameras aligned?

For each pair we compute edge maps (Sobel magnitude) of the visible and thermal images and find the shift
that best lines them up, by phase correlation. A perfectly registered pair gives a shift of (0, 0). Edges are
used, not raw intensities, because the two sensors see different things in the same place (a cold, bright
shirt; a hot, dark road), but object outlines appear in both.
'''),
        code('''
def edges(img):
    img = cv2.GaussianBlur(img.astype(np.float32), (5, 5), 0)
    return cv2.magnitude(cv2.Sobel(img, cv2.CV_32F, 1, 0), cv2.Sobel(img, cv2.CV_32F, 0, 1))

rng = np.random.default_rng(0)
fig, axes = plt.subplots(1, 2, figsize=(14, 3.6))
for ax, name in zip(axes, ("llvip", "m3fd")):
    files = sorted((env.data / name / "images" / "visible" / "val").glob("*.jpg"))
    shifts = []
    for f in rng.choice(files, size=min(200, len(files)), replace=False):
        v = edges(cv2.imread(str(f), cv2.IMREAD_GRAYSCALE))
        t = edges(cv2.imread(str(env.data / name / "images" / "infrared" / "val" / f.name), cv2.IMREAD_GRAYSCALE))
        win = cv2.createHanningWindow(v.shape[::-1], cv2.CV_32F)
        (dx, dy), resp = cv2.phaseCorrelate(v, t, win)
        if resp > 0.05:                       # ignore pairs with too little shared structure to measure
            shifts.append(np.hypot(dx, dy))
    shifts = np.array(shifts)
    ax.hist(shifts, bins=40, color="#3B8B66", alpha=0.85)
    ax.set_title(f"{name}: residual misalignment, {len(shifts)} pairs"); ax.set_xlabel("shift (px at 640)")
    audit[name]["misalignment_px"] = {q: float(np.percentile(shifts, q)) for q in (50, 90, 99)}
plt.tight_layout(); plt.show()
print({n: a["misalignment_px"] for n, a in audit.items()})
'''),
        md("**Reading it.** If the 90th percentile is a couple of pixels, the benchmark is registered and real "
           "misalignment cannot be studied on it: the synthetic shifts of 4, 8 and 16 px in notebook 09 then "
           "test robustness well beyond what the data contain. That limitation is stated in the pilot note."),
        code('''
(env.runs / "audit").mkdir(parents=True, exist_ok=True)
(env.runs / "audit" / "audit.json").write_text(json.dumps(audit, indent=2))
print(json.dumps(audit, indent=2))
'''),
    ])


def train_nb(path, title, purpose, notebook_id, runtime, extra_cells=(), optional_filter=False):
    cells = [
        header(title, purpose, "`cffm-net-pilot`, raw LLVIP and M3FD (or the output of 02)", runtime,
               "checkpoints in `runs/<run>/`, full-set metrics in `runs/eval/<run>/metrics.json`"),
        SETUP,
        md("## The runs\n\nStraight from `configs/pilot.yaml`. Batch is the total over both GPUs."),
    ]
    if optional_filter:
        cells.append(code(f'''
INCLUDE_OPTIONAL = False    # True also trains runs marked optional (e.g. the no-stride-4-head ablation)
specs = [s for s in pipeline.runs_for(plan, "{notebook_id}") if INCLUDE_OPTIONAL or not s.get("optional")]
''' + TRAIN_RUNS))
    else:
        cells.append(code(f'specs = pipeline.runs_for(plan, "{notebook_id}")\n' + TRAIN_RUNS))
    cells += [
        md("## Train and evaluate"),
        code(TRAIN_LOOP),
        md("## Learning curves"),
        code(CURVES),
        md("## Results on the full validation set"),
        code(SUMMARY),
        *extra_cells,
        SAVE_NOTE,
    ]
    write(path, cells)


def nb04():
    train_nb("notebooks/phase1_still_images/04_baselines_single_modality.ipynb",
             "04 · Single-sensor baselines",
             "What can one camera do alone? Stock YOLO26-n, fine-tuned from COCO, on LLVIP's visible images and, "
             "separately, on its thermal images. These two numbers set the bar every fusion model must clear: "
             "fusion that does not beat the better single sensor (at night, almost certainly thermal) is not "
             "worth its second backbone. This is the reference for hypothesis PH1.",
             "04", "about 1.5 to 2 hours on 2 x T4")


def nb05():
    train_nb("notebooks/phase1_still_images/05_baseline_two_stream.ipynb",
             "05 · Two-stream baseline",
             "The simplest fusion: two YOLO26-n backbones, their features concatenated and squeezed by a 1x1 "
             "convolution at strides 8, 16 and 32, and the standard YOLO26 head. Same initialisation, data, "
             "schedule and evaluation as CFFM-Net, so any difference later comes from the fusion design and the "
             "stride-4 pathway, not from training. Trained on LLVIP and on M3FD.",
             "05", "about 2 to 2.5 hours on 2 x T4")


def nb06():
    write("notebooks/phase1_still_images/06_inside_cmfm.ipynb", [
        header("06 · Inside the CMFM block",
               "No training here. We open up the Cross-Modal Fusion Mamba block and check, with tensors in "
               "hand, the properties the method claims: how big each part is, what a fresh block computes, that a "
               "zero-reliability sensor really cannot write into the scan's state, how the tokens are ordered, "
               "and that the gated-convolution control is matched to CFFM-Net in compute.",
               "`cffm-net-pilot`", "about 2 minutes", "nothing; this notebook is for understanding and checking"),
        SETUP,
        md('''
## 0. The method in three pictures

**CFFM-Net.** There are two YOLO26-n backbones, one per sensor, with separate weights; both start from COCO.
They are fused at four levels, from P2 (stride 4) to P5 (stride 32). P2 uses a reliability-weighted sum, and
P3 to P5 use CMFM. The fused maps feed YOLO26's stride-4 PAN neck and its NMS-free heads.
'''),
        figure("fig_architecture", width=700),
        md('''
**CMFM.** One reliability head scores each sensor at each location, and that score has two jobs: it gates
the scan and it weights the residual. Thermal features are first shifted by a learned offset field. The two
streams are then interleaved token by token and scanned in four directions. The output projection starts at
zero, so a fresh block is exactly the reliability-weighted average.
'''),
        figure("fig_cmfm", width=640),
        md(r'''
**The gated recurrence.** Each token's step $\Delta_k$ is multiplied by the reliability $r_k$ of its sensor.
As $r_k \to 0$, $\bar A_k \to 1$ and $\bar B_k \to 0$, so the token neither writes into the state nor
erases it. Section 3 below checks this numerically.
'''),
        figure("fig_gated_scan", width=640),
        md("## 1. Where the parameters go\n\nOne CMFM block at the stride-8 level of YOLO26-n (128 channels)."),
        code('''
import torch, pandas as pd
from cffm.blocks import CMFM, GatedCrossScan
torch.manual_seed(0)
blk = CMFM(128, impl="torch").eval()
rows = [{"part": n, "params": sum(p.numel() for p in m.parameters())} for n, m in blk.named_children()]
df = pd.DataFrame(rows); df.loc[len(df)] = {"part": "total", "params": df["params"].sum()}; df
'''),
        md("## 2. A fresh block is a reliability-weighted average\n\nThe output projection starts at zero and both "
           "reliabilities start at sigmoid(2) = 0.88, so before training CMFM returns exactly (F_v + F_t) / 2. "
           "Training can only *add* to that safe starting point."),
        code('''
fv, ft = torch.randn(2, 128, 40, 40), torch.randn(2, 128, 40, 40)
z = blk(fv, ft)
print("max |z - (fv + ft)/2| =", (z - (fv + ft) / 2).abs().max().item())
'''),
        md('''
## 3. The property everything rests on

Set the thermal reliability to zero. Every thermal token then gets a step of zero, so it can neither write
into the scan's state nor erase it, and the outputs at visible positions must not change however much we
perturb the thermal features. With reliability one, the same perturbation must change them.
'''),
        code('''
scan = GatedCrossScan(16, directions=4, impl="torch").eval()
xv, xt = torch.randn(1, 16, 12, 12), torch.randn(1, 16, 12, 12)
ones, zeros = torch.ones(1, 1, 12, 12), torch.zeros(1, 1, 12, 12)
noise = 5 * torch.randn_like(xt)
d_off = (scan(xv, xt, ones, zeros)[0] - scan(xv, xt + noise, ones, zeros)[0]).abs().max().item()
d_on = (scan(xv, xt, ones, ones)[0] - scan(xv, xt + noise, ones, ones)[0]).abs().max().item()
print(f"r_thermal = 0: visible outputs move by {d_off:.2e}   (should be ~0)")
print(f"r_thermal = 1: visible outputs move by {d_on:.2e}   (should be clearly > 0)")
'''),
        md("## 4. Token order\n\nThe first scan direction on a 2 x 3 map, written as (row, column, modality). "
           "Visible (v) and thermal (t) alternate at every location: fusion happens inside the state."),
        code('''
X = torch.zeros(1, 1, 2, 3, 2)
for r in range(2):
    for c in range(3):
        X[0, 0, r, c, 0], X[0, 0, r, c, 1] = 10 * r + c, 100 + 10 * r + c
seq = GatedCrossScan(1, directions=4)._orders(X)[0, 0, 0].int().tolist()
print(" ".join(f"{'t' if s >= 100 else 'v'}({(s % 100) // 10},{s % 10})" for s in seq))
'''),
        md("## 5. Compute of every model, and the matched control\n\nGFLOPs at 640 x 640 include the element-wise "
           "work inside the scans (profilers do not count it; see `cffm.train.gflops`). The gated-convolution "
           "control (C4) should sit within a few percent of CFFM-Net."),
        code('''
from cffm.model import DualStreamDetectionModel
from cffm.train import gflops, scan_gflops
rows = []
for cfg in sorted((env.project / "configs" / "models").glob("*.yaml")):
    m = DualStreamDetectionModel(str(cfg), nc=1, verbose=False)
    rows.append({"model": cfg.stem, "params (M)": round(sum(p.numel() for p in m.parameters()) / 1e6, 3),
                 "GFLOPs": round(gflops(m, 640), 2), "of which scans": round(scan_gflops(m, 640), 3)})
df = pd.DataFrame(rows).set_index("model"); print(df)
ratio = df.loc["cffm-net-n-gconv", "GFLOPs"] / df.loc["cffm-net-n", "GFLOPs"]
print(f"control / CFFM-Net GFLOPs = {ratio:.3f}"); assert abs(ratio - 1) < 0.05
'''),
    ])


def nb07():
    extra = [
        md("## What it sees\n\nPredictions (green) against ground truth (red) on night scenes from the validation "
           "set, then the reliability maps at stride 8: where the model trusts visible, and where thermal. In a "
           "dark scene, thermal reliability should be high on people and visible reliability should drop."),
        code('''
import matplotlib.pyplot as plt
from cffm.train import _load
from cffm.viz import draw_pair, plot_reliability, predict_pair, reliability_maps

model = _load(env.runs / "llvip_cffm" / "weights" / "best.pt").to("cuda:0")
names = model.names
val = sorted((env.data / "llvip" / "images" / "visible" / "val").glob("*.jpg"))
for v in val[200:2000:450]:
    det, pair = predict_pair(model, str(v), conf=0.3)
    h, w = pair.shape[:2]; gt = []
    for line in (env.data / "llvip" / "labels" / "visible" / "val" / (v.stem + ".txt")).read_text().splitlines():
        c, x, y, bw, bh = map(float, line.split())
        gt.append([(x - bw / 2) * w, (y - bh / 2) * h, (x + bw / 2) * w, (y + bh / 2) * h])
    plt.figure(figsize=(14, 4.5)); plt.imshow(draw_pair(pair, det, names, gt)); plt.axis("off"); plt.title(v.name)
    plt.show()
'''),
        code('''
for v in val[300:1500:600]:
    maps, lb = reliability_maps(model, str(v))
    plot_reliability(maps, lb, level=3); plt.show()
'''),
    ]
    train_nb("notebooks/phase1_still_images/07_train_cffm_net.ipynb",
             "07 · Train CFFM-Net",
             "The method itself, in image mode: two YOLO26-n backbones, a reliability-weighted sum at stride 4, "
             "CMFM blocks (reliability-gated cross-modal selective scan, offset alignment, local branch) at "
             "strides 8, 16 and 32, and the YOLO26 neck with a stride-4 detection head. Trained on LLVIP and on "
             "M3FD with exactly the settings of the baselines.",
             "07", "about 3 to 3.5 hours on 2 x T4", extra_cells=extra)


def nb08():
    train_nb("notebooks/phase1_still_images/08_ablations.ipynb",
             "08 · Ablations",
             "Each run changes one thing in CFFM-Net, so a difference in results can only come from that one "
             "thing. **Gating off** (r = 1 everywhere) tests whether reliability gating adds value (C1, PH2). "
             "**Gated convolution** replaces the selective scan with a gated depthwise convolution of matched "
             "compute, the MambaOut-style control (C4, PH4). **No stride-4 head** (optional) isolates the "
             "small-object pathway (C2).",
             "08", "about 4 to 5 hours on 2 x T4 (about 6 with the optional run)", optional_filter=True)


def nb09():
    write("notebooks/phase1_still_images/09_probe_and_latency.ipynb", [
        header("09 · Degradation probe and latency",
               "Two measurements that need finished checkpoints but no training. **The probe** asks what happens "
               "when a sensor misbehaves at test time: the visible image goes dark or black, the thermal image "
               "goes black, or the thermal image shifts by 4, 8 or 16 pixels. A model whose fusion is truly "
               "selective should lose less (PH2). **Latency** measures every model on one T4 at batch 1 in FP16, "
               "with parameters and GFLOPs (PH5, contribution C5).",
               "`cffm-net-pilot`, raw LLVIP (or the output of 02), and the **saved outputs of notebooks 04, 05, 07 "
               "and 08** (*Add Input → Your Work*)",
               "about 1 to 1.5 hours on 1 T4", "`runs/probe/*/metrics.json`, `runs/latency/latency.json`"),
        SETUP,
        code('''
pipeline.prepare("llvip", env, plan)
ckpts = pipeline.find_checkpoints(env)
for k, v in sorted(ckpts.items()):
    print(f"{k:24s} {v}")
missing = [r for r in plan["probe"]["runs"] if r not in ckpts]
assert not missing, f"attach the saved outputs of the notebooks that trained {missing}"
'''),
        md('''
## 1. The probe

Each degradation is applied to the input batch inside the model's forward pass, with a fixed seed, so every
model sees identical corrupted images. *Flagged* drops also tell the model which sensor is missing (as a
deployed system would know); unflagged ones make it work that out from the images alone. Only CFFM-Net can use
the flag; the others must cope.
'''),
        code('''
import pandas as pd
from cffm.train import probe

data = env.data / "llvip" / "data_paired.yaml"
rows = []
for run in plan["probe"]["runs"]:
    for kind in plan["probe"]["kinds"]:
        for flagged in ([False, True] if kind in plan["probe"]["flagged"] else [False]):
            m = probe(ckpts[run], data, kind, flagged=flagged, device=0, project=str(env.runs / "probe"))
            rows.append({"run": run, "probe": kind + (" (flagged)" if flagged else ""),
                         "AP": m["mAP50-95(B)"], "AP50": m["mAP50(B)"]})
            print(f"{run:20s} {rows[-1]['probe']:26s} AP {m['mAP50-95(B)']:.4f}")
probe_df = pd.DataFrame(rows)
probe_df.to_csv(env.runs / "probe" / "probe.csv", index=False)
'''),
        code('''
import matplotlib.pyplot as plt
piv = probe_df.pivot(index="probe", columns="run", values="AP").loc[probe_df["probe"].unique()]
ax = piv.plot(kind="barh", figsize=(10, 6), width=0.8)
ax.set_xlabel("COCO AP50-95 on LLVIP (full validation set)"); ax.invert_yaxis(); ax.grid(alpha=0.3, axis="x")
plt.tight_layout(); plt.show()
'''),
        md("## 2. Parameters, GFLOPs and latency\n\nAll on GPU 0, FP16, batch 1, a 640 x 512 input (LLVIP's shape), "
           "50 warm-up passes then 300 timed ones with a short pause between them so the GPU does not throttle. "
           "This times the PyTorch model; the full protocol of the dossier adds TensorRT export (Phase 3)."),
        code('''
import json, torch
from cffm.train import _load, gflops, latency_ms
lat = {}
for run, w in sorted(ckpts.items()):
    m = _load(w)
    dual = getattr(m, "_dual", False)
    g = gflops(m, 640)
    t = latency_ms(m, imgsz=(512, 640), channels=4 if dual else 3, device="cuda:0", half=True, pause_s=0.02)
    lat[run] = {"params_M": sum(p.numel() for p in m.parameters()) / 1e6, "GFLOPs": g, **t}
    print(f"{run:24s} {lat[run]['params_M']:6.2f} M  {g:6.2f} GFLOPs  {t['median_ms']:6.2f} ms (p95 {t['p95_ms']:.2f})")
    del m; torch.cuda.empty_cache()
(env.runs / "latency").mkdir(parents=True, exist_ok=True)
(env.runs / "latency" / "latency.json").write_text(json.dumps(lat, indent=2))
'''),
        SAVE_NOTE,
    ])


def nb15():
    write("notebooks/phase1_still_images/15_results_and_figures.ipynb", [
        header("15 · Results, hypotheses and figures",
               "The only route from experiments to the write-up. It reads every logged metric, builds the results "
               "tables, decides each pilot hypothesis with the rule fixed in advance in the pilot note, and draws "
               "the figures. No number is typed by hand: if a run changes, rerun this notebook and every table "
               "changes with it. The decision rules live in `cffm/report.py`, where they are unit-tested.",
               "`cffm-net-pilot` and the **saved outputs of notebooks 04, 05, 07, 08 and 09**",
               "a few minutes", "`results/` with tables (Markdown, LaTeX, CSV) and figures (PNG, PDF)"),
        SETUP,
        code('''
import json
from cffm import report
ev = pipeline.find_metrics(env, "eval")       # full-validation-set metrics of every trained run
pr = pipeline.find_metrics(env, "probe")      # metrics under each test-time degradation
lat = {}
for root in [env.runs, *[r for r in pipeline.INPUT_ROOTS if r.exists()]]:
    for f in root.glob("**/latency/latency.json"):
        lat.update({k: v for k, v in json.loads(f.read_text()).items() if k not in lat})
print(len(ev), "evaluated runs,", len(pr), "probe results,", len(lat), "latency entries")
out = env.project / "results"; out.mkdir(exist_ok=True)
'''),
        md("## 1. Main results\n\nAP is COCO AP50-95 on the full validation set. AP_vt / AP_t / AP_s are the AI-TOD "
           "bands (<8, 8-16, 16-32 px); AP_small is COCO's <32 px. Cost is measured on one T4, FP16, batch 1."),
        code('''
main = report.main_table(ev, lat)
main.round(4)
'''),
        md('''
## 2. The hypotheses, decided by the rules written down before training

From the pilot note (Table 4). Thresholds are in AP points (0.015 = 1.5 points). A difference inside the
threshold is **inconclusive**, not a win: with one seed per configuration, run-to-run variation is about 1 to
1.5 points. *Against* means the difference is as large as the threshold in the wrong direction.
'''),
        code('''
hyp = report.decide(ev, pr, lat, plan["probe"]["kinds"])
hyp
'''),
        md("## 3. Figures"),
        code('''
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "serif", "axes.spines.top": False, "axes.spines.right": False})

llvip = main[main.index.str.startswith("llvip")]
fig, ax = plt.subplots(figsize=(8, 3.8))
ax.barh(llvip.index, llvip["AP"], color=["#3B8B66" if "cffm" in i else "#66707B" for i in llvip.index])
ax.set_xlabel("COCO AP50-95, LLVIP (full validation set)"); ax.invert_yaxis(); ax.grid(axis="x", alpha=0.3)
plt.tight_layout(); plt.savefig(out / "fig_llvip_ap.png", dpi=200); plt.savefig(out / "fig_llvip_ap.pdf"); plt.show()

m3 = main.loc[[r for r in ("m3fd_concat", "m3fd_cffm") if r in main.index], ["AP_vt", "AP_t", "AP_s", "AP"]]
if len(m3):
    ax = m3.T.plot(kind="bar", figsize=(7, 3.6), color=["#66707B", "#3B8B66"], rot=0)
    ax.set_ylabel("AP50-95"); ax.set_title("M3FD by object size"); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout(); plt.savefig(out / "fig_m3fd_sizes.png", dpi=200); plt.savefig(out / "fig_m3fd_sizes.pdf"); plt.show()
'''),
        code('''
probe = report.probe_table(pr, plan["probe"]["runs"], plan["probe"]["kinds"], plan["probe"]["flagged"])
if len(probe):
    ax = probe.plot(kind="barh", figsize=(9, 5.5), width=0.8)
    ax.set_xlabel("COCO AP50-95 under test-time degradation (LLVIP)"); ax.invert_yaxis(); ax.grid(axis="x", alpha=0.3)
    plt.tight_layout(); plt.savefig(out / "fig_probe.png", dpi=200); plt.savefig(out / "fig_probe.pdf"); plt.show()
probe.round(4)
'''),
        md("## 4. Tables for the write-up\n\nMarkdown for notes, CSV for spreadsheets, and LaTeX in the dossier's "
           "rule-free style for the paper."),
        code('''
main.round(4).to_csv(out / "main_results.csv")
hyp.to_csv(out / "hypotheses.csv")
if len(probe): probe.round(4).to_csv(out / "probe.csv")
(out / "tables.md").write_text("## Main results\\n\\n" + report.to_markdown(main) +
                               "\\n## Hypotheses\\n\\n" + report.to_markdown(hyp) +
                               ("\\n## Degradation probe\\n\\n" + report.to_markdown(probe) if len(probe) else ""))
(out / "tables.tex").write_text(report.to_latex(main, "Pilot results, one seed per configuration.", "tab:pilot")
                                + "\\n" + report.to_latex(hyp, "Pilot hypotheses and verdicts.", "tab:pilot-hyp"))
print(sorted(p.name for p in out.iterdir()))
'''),
        SAVE_NOTE,
    ])

if __name__ == "__main__":
    for f in (nb00, nb01, nb02, nb03, nb04, nb05, nb06, nb07, nb08, nb09, nb15):
        f()
