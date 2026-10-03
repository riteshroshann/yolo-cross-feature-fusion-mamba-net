"""Generate every pilot notebook from one script, so the shared cells never drift apart.

    python tools/build_notebooks.py
"""
import io
import re
import tokenize
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]


def md(text):
    return nbf.v4.new_markdown_cell(text.strip("\n"))


def uncomment(src):
    """Drop every comment; lines that held only a comment disappear."""
    cuts = {t.start[0] - 1: t.start[1] for t in tokenize.generate_tokens(io.StringIO(src).readline)
            if t.type == tokenize.COMMENT}
    out = []
    for i, line in enumerate(src.splitlines()):
        if i in cuts:
            if not line[:cuts[i]].strip():
                continue
            line = line[:cuts[i]].rstrip()
        out.append(line)
    return re.sub(r"\n{3,}", "\n\n", "\n".join(out))


def code(text):
    return nbf.v4.new_code_cell(uncomment(text.strip("\n")))


def write(path, cells):
    nb = nbf.v4.new_notebook()
    nb.cells = cells
    nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}}
    out = ROOT / path
    out.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, out)
    print("wrote", path)


SETUP = code('''
import glob, shutil, subprocess, sys
from pathlib import Path

ON_KAGGLE = Path("/kaggle/working").exists()
if ON_KAGGLE:
    PROJECT = Path("/kaggle/working/cffm-net-pilot")
    if not PROJECT.exists():
        hits = [Path(p).parent for k in range(1, 6) for p in glob.glob("/kaggle/input" + "/*" * k + "/pyproject.toml")
                if (Path(p).parent / "src" / "cffm").is_dir()]
        hits.sort(key=lambda h: any((h.parent / d).exists() for d in ("runs", "data")))
        zips = [Path(p) for k in range(1, 5) for p in glob.glob("/kaggle/input" + "/*" * k + ".zip")
                if "cffm" in Path(p).name]
        if hits:
            shutil.copytree(hits[0], PROJECT, copy_function=shutil.copyfile)
        elif zips:                       # the zip was attached but not unpacked by Kaggle
            shutil.unpack_archive(str(zips[0]), "/kaggle/working")   # it holds one folder, cffm-net-pilot/
        assert PROJECT.exists(), "Add the 'cffm-net-pilot' dataset (the uploaded zip) as an input to this notebook."
        for p in [PROJECT, *PROJECT.rglob("*")]:     # inputs are read-only; our copy must be writable
            p.chmod(p.stat().st_mode | 0o200)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ultralytics==8.4.171",
                    "faster-coco-eval>=1.6.7", "cloudpickle", "pytest", "lap>=0.5.12", "imageio-ffmpeg"], check=True)   # cloudpickle: two-GPU launcher
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "-e", str(PROJECT)], check=True)
else:
    PROJECT = next(p for p in [Path.cwd().resolve(), *Path.cwd().resolve().parents] if (p / "src" / "cffm").is_dir())
sys.path.insert(0, str(PROJECT / "src"))

import cffm
from cffm import env as cenv, pipeline
print("cffm", cffm.__version__, "imported from", Path(cffm.__file__).parent)   # must be inside PROJECT
env = cenv.setup()                       # also moves into the project folder
plan = pipeline.load_plan(env.project)   # configs/pilot.yaml
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
results = [pipeline.train_from_spec(s, env) for s in specs]
'''

CURVES = '''
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
import pandas as pd
keys = ["mAP50-95(B)", "mAP50(B)", "mAP_small(B)", "AP_vt(B)", "AP_t(B)", "AP_s(B)", "AP_m(B)", "AP_l(B)"]
rows = [{"run": r["run"], **{k.replace("(B)", ""): r["metrics"].get(k) for k in keys}} for r in results]
pd.DataFrame(rows).set_index("run").round(4)
'''

CLEANUP = code('''
if env.platform == "kaggle":
    shutil.rmtree(env.data, ignore_errors=True)
''')

SAVE_NOTE = md('''
### Keep the results

On Kaggle, click **Save Version → Save & Run All (Commit)** so `runs/` becomes an output that notebooks **09**
and **15** can attach (*Add Input → Your Work → this notebook*). Without a saved version the checkpoints vanish.
''')


def nb00():
    write("notebooks/phase0_environment/00_environment_setup.ipynb", [
        header("00 · Environment setup and checks",
               "Phase 0 of the CFFM-Net pilot: before any training, prove that every moving part works on this "
               "exact machine. If something is broken, this is the cheapest place to find out.",
               "`cffm-net-pilot` (the uploaded zip)", "about 10 to 15 minutes",
               "a go / no-go answer, measured scan speed and memory, and `weights/yolo26n.pt`"),
        SETUP,
        md("## 0. The pilot at a glance\n\nEleven notebooks in two phases; solid arrows carry data and checkpoints, "
           "dashed arrows metrics. The training notebooks (04, 05, 07, 08) are separate Kaggle sessions whose "
           "saved outputs 09 and 15 read."),
        figure("fig_workflow", width=900),
        md("## 1. What are we running on?"),
        code('''
import json, torch
print(json.dumps(cenv.versions(), indent=2))
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f"GPU {i}: {p.name}, {p.total_memory / 2**30:.1f} GB, compute capability {p.major}.{p.minor}")
assert torch.cuda.is_available(), "No GPU: set Accelerator to 'GPU T4 x2' in the notebook settings."
'''),
        md("## 2. COCO-pretrained YOLO26-n weights\n\nEvery model starts from these weights, so one copy goes "
           "in `weights/` and every run uses the identical file."),
        code('''
import torch
w = env.weights / "yolo26n.pt"
if not w.exists():
    torch.hub.download_url_to_file("https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt", str(w))
print(w, f"{w.stat().st_size / 1e6:.1f} MB")
'''),
        md('''
## 3. (Optional) the fused Mamba kernel

The pilot does **not** need it: the tested pure-PyTorch scan runs anywhere, and `mamba_ssm` takes 20+ minutes
to compile and may not support the T4. Leave the flag `False` unless you want to try; if it installs, restart
the kernel and it is picked up automatically.
'''),
        code('''
TRY_FUSED_SCAN = False
if TRY_FUSED_SCAN:
    r = subprocess.run([sys.executable, "-m", "pip", "install", "--no-build-isolation", "causal-conv1d", "mamba-ssm"],
                       capture_output=True, text=True)
    print(r.stdout[-3000:], r.stderr[-3000:])
    print("Now restart the kernel and run the notebook again from the top.")
from cffm import scan
print("mamba_ssm kernel:", scan.HAS_MAMBA_SSM, "| Triton kernel:", scan.HAS_TRITON)
'''),
        md('''
## 4. Unit tests

Small tests, each checking one claim the method depends on. Two need the optional fused kernel and four are
an opt-in smoke run (section 7 does that run itself), so expect 6 skipped.
'''),
        code('''
r = subprocess.run([sys.executable, "-m", "pytest", "-q", "tests"], cwd=env.project, capture_output=True, text=True)
print(r.stdout[-4000:])
assert r.returncode == 0, r.stderr[-4000:]
'''),
        md('''
## 5. How fast is the scan here?

Forward and backward at the size of CFFM-Net's biggest scan: stride 8 on a 640 crop, so **12 800 interleaved
tokens**, **256 scan channels** and **8 images per GPU**.
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

for impl, ok in (("torch", True), ("triton", scan.HAS_TRITON), ("cuda", scan.HAS_MAMBA_SSM)):
    if ok:
        ms, gb = time_scan(impl=impl)
        print(f"{impl:6s} scan, stride-8 size, batch 8: {ms:7.1f} ms forward+backward, peak {gb:.2f} GB")
'''),
        md('''
## 6. Do the planned batch sizes fit?

One mixed-precision forward and backward pass per model at its planned batch per GPU, measuring only memory and
step time. The last column projects one 30-epoch run on LLVIP's ~4 000 training pairs.
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
fits = (df["peak GB"] < limit - 1.0).all()
if env.platform == "kaggle":
    assert fits, "A batch is too large for this GPU: lower it in configs/pilot.yaml."
elif not fits:
    print(f"Some planned batches exceed this {limit:.0f} GB GPU; they are sized for a 15 GB T4.")
'''),
        md('''
## 7. The whole pipeline, end to end, on fake data and both GPUs

One epoch per model on a tiny synthetic dataset, through the real trainer, two-GPU launcher, validator and
probe. The numbers mean nothing; the point is that no step crashes.
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

If every cell above ran without an error, the environment is ready. Clean up the smoke runs, then continue with
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
               "Gets **LLVIP** (15 488 pedestrian pairs, mostly at night) and **M3FD** (4 200 pairs, six classes) "
               "onto the machine and checks they are complete. Both are for non-commercial research only, so do "
               "not make your Kaggle copies public.",
               "`cffm-net-pilot`, plus LLVIP and M3FD (see below)", "a few minutes (longer if downloading)",
               "verified raw datasets"),
        md('''
## How to get the data onto Kaggle

**Option A (recommended).** Download the official archives on your computer and upload each as a **private**
Kaggle dataset (*Datasets → New Dataset → upload the zip*; Kaggle unzips it). Then attach both here
(*Add Input → Your Work*).

* LLVIP: <https://bupt-ai-cz.github.io/LLVIP/>, with the February 2023 corrected annotations.
* M3FD: the *M3FD_Detection* part (`Vis/`, `Ir/`, `Annotation/`) of <https://github.com/JinyuanLiu-CV/TarDAL>.

**Option B.** Attach a public Kaggle copy (search for "LLVIP" / "M3FD") with the official folder layout and
the corrected LLVIP labels.

**Option C.** Paste a public Google Drive link or file ID below and download inside this notebook.

Datasets are found by folder layout, so their names do not matter.
'''),
        SETUP,
        code('''
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
        md("## Are they complete?\n\nCounts per folder, complete pairs, and one parsed annotation to check the "
           "class names."),
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
        md("## One pair from each\n\nIf the two images do not show the same scene, the pairing is wrong."),
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
               "Converts the raw datasets into a paired layout Ultralytics can read: same-named visible and "
               "infrared images at 640 px, YOLO labels and three data YAMLs per dataset. Every choice is recorded "
               "in `dataset_card.json`.",
               "`cffm-net-pilot`, raw LLVIP and M3FD", "about 5 minutes",
               "`data/llvip/`, `data/m3fd/` (and, if you save a version, an attachable output)"),
        SETUP,
        md("## Convert\n\nOptions come from `configs/pilot.yaml`, so every notebook builds the identical dataset."),
        code('''
cards = {}
for name in ("llvip", "m3fd"):
    out = pipeline.prepare(name, env, plan, force=False)
    cards[name] = __import__("json").loads((out / "dataset_card.json").read_text())
    print(name, "->", out)
'''),
        md("## What we made\n\nThe object-size bins show how small the objects are at training resolution."),
        code('''
import pandas as pd
rows = [{"dataset": n, "train": c["images"]["train"], "val (full)": c["images"]["val"], "objects": c["objects"],
         **c["object_size_bins"]} for n, c in cards.items()]
pd.DataFrame(rows).set_index("dataset")
'''),
        code('''
print((env.data / "llvip" / "data_paired.yaml").read_text())
'''),
        md("## Sanity checks\n\nEvery visible image has a same-size thermal partner and a label file, and every "
           "box lies inside the image."),
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
        size_ok = all(cv2.imread(str(v)).shape[:2] == cv2.imread(str(root / "images" / "infrared" / split / v.name)).shape[:2]
                      for v in vis[::25])
        print(f"{name} {split}: {len(vis)} pairs, {len(missing)} incomplete, {len(boxes)} boxes, "
              f"all inside image: {bool(inside)}, pair sizes match: {size_ok}")
        assert vis and not missing and inside and size_ok
'''),
        md("## Look at a few converted pairs, with their labels\n\nA box beside the person in the thermal image "
           "means misregistration (notebook 03 measures it)."),
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


def nb03():
    write("notebooks/phase1_still_images/03_data_audit.ipynb", [
        header("03 · Data audit",
               "How small are the objects, how dark are the scenes, and how well are the two cameras aligned? "
               "The alignment result also sets the synthetic shifts of the probe in notebook 09.",
               "`cffm-net-pilot`, raw LLVIP and M3FD (or the output of 02)", "about 5 minutes",
               "`runs/audit/audit.json` and the figures below"),
        SETUP,
        code('''
for name in ("llvip", "m3fd"):
    pipeline.prepare(name, env, plan)
'''),
        md("## 1. Object sizes at training resolution\n\nSize is the square root of box area at 640 px; dashed "
           "lines mark the AI-TOD bands (8, 16, 32 px) and COCO's 96 px boundary."),
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
        md("## 2. How dark are the scenes?\n\nMean visible brightness; LLVIP is mostly night, where thermal should "
           "earn its place."),
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

Phase correlation of Sobel edge maps gives the shift that best lines up each pair, (0, 0) if registered.
Edges are used because outlines appear in both sensors even where intensities differ.
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
        if resp > 0.05:                       # too little shared structure to measure
            shifts.append(np.hypot(dx, dy))
    shifts = np.array(shifts)
    ax.hist(shifts, bins=40, color="#3B8B66", alpha=0.85)
    ax.set_title(f"{name}: residual misalignment, {len(shifts)} pairs"); ax.set_xlabel("shift (px at 640)")
    audit[name]["misalignment_px"] = {q: float(np.percentile(shifts, q)) for q in (50, 90, 99)}
plt.tight_layout(); plt.show()
print({n: a["misalignment_px"] for n, a in audit.items()})
'''),
        md("**Reading it.** If the 90th percentile is a couple of pixels, the benchmark is registered, and the "
           "4, 8 and 16 px shifts of notebook 09 test robustness well beyond the data."),
        code('''
(env.runs / "audit").mkdir(parents=True, exist_ok=True)
(env.runs / "audit" / "audit.json").write_text(json.dumps(audit, indent=2))
print(json.dumps(audit, indent=2))
'''),
        md("## 4. What the labels look like\n\nThe four most crowded test scenes of each dataset with their human "
           "labels, visible and thermal side by side."),
        code('''
from cffm import hud, showcase as sc
from cffm.viz import make_pair
for name in ("llvip", "m3fd"):
    names = json.loads((env.data / name / "dataset_card.json").read_text())["names"]
    boards = []
    for stem in sc.crowded(env.data / name, k=4):
        pair = make_pair(*sc.pair_paths(env, name, stem))
        gt = sc.gt_boxes(env, name, stem, pair.shape)
        boards.append(hud.render(pair, gt, names, panels=("visible", "thermal"), show_conf=False,
                                 title=name.upper(), subtitle=f"GROUND TRUTH · {len(gt)} LABELS", frame=int(stem) % 10000))
    sc.show(hud.grid(boards, cols=2))
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
INCLUDE_OPTIONAL = False    # True also trains optional runs (the no-stride-4-head ablation)
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
        CLEANUP,
        SAVE_NOTE,
    ]
    write(path, cells)


def nb04():
    train_nb("notebooks/phase1_still_images/04_baselines_single_modality.ipynb",
             "04 · Single-sensor baselines",
             "Stock YOLO26-n, fine-tuned from COCO, on LLVIP's visible images and separately on its thermal "
             "images. Fusion must beat the better of the two to be worth its second backbone (PH1).",
             "04", "about 1.5 to 2 hours on 2 x T4")


def nb05():
    train_nb("notebooks/phase1_still_images/05_baseline_two_stream.ipynb",
             "05 · Two-stream baseline",
             "The simplest fusion: two YOLO26-n backbones, features concatenated and squeezed by a 1x1 conv at "
             "strides 8, 16 and 32, and the standard head. Everything else matches CFFM-Net, on LLVIP and M3FD.",
             "05", "about 2 to 2.5 hours on 2 x T4")


def nb06():
    write("notebooks/phase1_still_images/06_inside_cmfm.ipynb", [
        header("06 · Inside the CMFM block",
               "No training: we open the Cross-Modal Fusion Mamba block and check its claimed properties with "
               "tensors in hand.",
               "`cffm-net-pilot`", "about 2 minutes", "nothing; this notebook is for understanding and checking"),
        SETUP,
        md('''
## 0. The method in three pictures

**CFFM-Net.** Two YOLO26-n backbones, one per sensor, fused at P2 by a reliability-weighted sum and at P3 to
P5 by CMFM. The fused maps feed YOLO26's stride-4 PAN neck and its NMS-free heads.
'''),
        figure("fig_architecture", width=700),
        md('''
**CMFM.** A per-location reliability score gates the scan and weights the residual, after thermal is aligned
by a learned offset. The two streams are interleaved token by token and scanned in four directions.
'''),
        figure("fig_cmfm", width=640),
        md(r'''
**The gated recurrence.** Each token's step $\Delta_k$ is multiplied by the reliability $r_k$ of its sensor.
As $r_k \to 0$, $\bar A_k \to 1$ and $\bar B_k \to 0$, so the token neither writes into the state nor
erases it.
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
           "reliabilities at sigmoid(2), so before training CMFM returns exactly (F_v + F_t) / 2."),
        code('''
fv, ft = torch.randn(2, 128, 40, 40), torch.randn(2, 128, 40, 40)
z = blk(fv, ft)
print("max |z - (fv + ft)/2| =", (z - (fv + ft) / 2).abs().max().item())
'''),
        md('''
## 3. The property everything rests on

With thermal reliability zero, perturbing the thermal features must not move the visible outputs at all; with
reliability one, it must.
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
        md("## 4. Token order\n\nThe first scan direction on a 2 x 3 map as (row, column, modality): visible (v) "
           "and thermal (t) alternate at every location."),
        code('''
X = torch.zeros(1, 1, 2, 3, 2)
for r in range(2):
    for c in range(3):
        X[0, 0, r, c, 0], X[0, 0, r, c, 1] = 10 * r + c, 100 + 10 * r + c
seq = GatedCrossScan(1, directions=4)._orders(X)[0, 0, 0].int().tolist()
print(" ".join(f"{'t' if s >= 100 else 'v'}({(s % 100) // 10},{s % 10})" for s in seq))
'''),
        md("## 5. Compute of every model, and the matched control\n\nGFLOPs at 640 x 640 include the scans' "
           "element-wise work; the gated-convolution control (C4) should sit within a few percent of CFFM-Net."),
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
        md("## What it sees\n\nThe model's detections on night scenes, with the thermal view and the fusion trust "
           "map (blue: visible weighted, orange: thermal weighted), and a log of every subject."),
        code('''
from IPython.display import display
from cffm import hud, showcase as sc
from cffm.train import _load
model = _load(env.runs / "llvip_cffm" / "weights" / "best.pt").to("cuda:0")
for stem in sc.crowded(env.data / "llvip", k=3):
    a = hud.analyze(model, *sc.pair_paths(env, "llvip", stem), frame=int(stem) % 10000)
    sc.show(a.board)
    display(sc.subject_log(a.det, a.names, a.share))
'''),
        code('''
model = _load(env.runs / "m3fd_cffm" / "weights" / "best.pt").to("cuda:0")
for stem in sc.crowded(env.data / "m3fd", k=2):
    sc.show(hud.analyze(model, *sc.pair_paths(env, "m3fd", stem), frame=int(stem) % 10000).board)
'''),
    ]
    train_nb("notebooks/phase1_still_images/07_train_cffm_net.ipynb",
             "07 · Train CFFM-Net",
             "The method in image mode: two YOLO26-n backbones, a reliability-weighted sum at stride 4, CMFM at "
             "strides 8 to 32 and a stride-4 detection head. Trained on LLVIP and M3FD with the baselines' settings.",
             "07", "about 3 to 3.5 hours on 2 x T4", extra_cells=extra)


def nb08():
    train_nb("notebooks/phase1_still_images/08_ablations.ipynb",
             "08 · Ablations",
             "Each run changes one thing in CFFM-Net: **gating off** (C1, PH2), a **gated convolution** of "
             "matched compute instead of the scan (C4, PH4), and optionally **no stride-4 head** (C2).",
             "08", "about 4 to 5 hours on 2 x T4 (about 6 with the optional run)", optional_filter=True)


def nb08b():
    train_nb("notebooks/phase1_still_images/08b_head_control.ipynb",
             "08b · Head control",
             "CFFM-Net uses YOLO26's stride-4 head and the concat baseline does not. Here concat gets the stride-4 "
             "head and CFFM-Net the standard one, so fusion and head can be compared separately.",
             "08b", "about 1.5 hours on 2 x T4")



def nb08c():
    train_nb("notebooks/phase1_still_images/08c_round2_llvip.ipynb",
             "08c · Round 2 on LLVIP: standard head and modality dropout",
             "CFFM-Net v2 drops the stride-4 head, which cost 1.8 AP on LLVIP, and trains with modality dropout: "
             "each image loses its visible or thermal camera with probability 0.15, flagged half of the time. The "
             "concat baseline gets the same augmentation, so the comparison stays fair.",
             "08c", "about 1.5 hours on 2 x T4")


def nb08d():
    train_nb("notebooks/phase1_still_images/08d_round2_m3fd.ipynb",
             "08d · Round 2 on M3FD: standard head and modality dropout",
             "The same pair of runs as 08c on M3FD's six classes and much smaller objects.",
             "08d", "about 1.5 hours on 2 x T4")



def nb08e():
    train_nb("notebooks/phase1_still_images/08e_round3_llvip.ipynb",
             "08e · Round 3 on LLVIP: 80 epochs",
             "CFFM-Net v2 and concat, both with modality dropout, on the long schedule: 80 epochs, mosaic off "
             "for the last 10.",
             "08e", "about 4 hours on 2 x T4")


def nb08f():
    train_nb("notebooks/phase1_still_images/08f_round3_m3fd.ipynb",
             "08f · Round 3 on M3FD: 80 epochs",
             "The same pair of runs as 08e on M3FD.",
             "08f", "about 4 hours on 2 x T4")


def nb09():
    write("notebooks/phase1_still_images/09_probe_and_latency.ipynb", [
        header("09 · Degradation probe and latency",
               "**The probe** measures how much each model loses when a sensor goes dark, black or misaligned at "
               "test time (PH2). **Latency** times every model on one T4 at batch 1 in FP16, with parameters and "
               "GFLOPs (PH5).",
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
        md("## 0. What the probe does to the input\n\nOne test pair under each degradation, exactly as the model "
           "receives it: the visible camera darkened or dropped, the thermal camera dropped or shifted."),
        code('''
import cv2, numpy as np
from cffm import hud, showcase as sc
from cffm.viz import make_pair
stem = sc.crowded(env.data / "llvip", k=2)[1]
pair = make_pair(*sc.pair_paths(env, "llvip", stem))
tiles = []
for kind in ("clean", "visible_dark", "visible_drop", "thermal_drop", "thermal_shift"):
    p = pair if kind == "clean" else sc.degrade(pair, kind)
    v = cv2.cvtColor(hud.enhance(p[..., :3])[0] if kind == "clean" else p[..., :3], cv2.COLOR_BGR2RGB)
    t = hud.thermal_rgb(p[..., 3]) if p[..., 3].any() else np.zeros_like(v)
    tile = np.concatenate([v, t], 0)
    cv2.putText(tile, kind.upper(), (20, 60), cv2.FONT_HERSHEY_DUPLEX, 1.6, (255, 204, 0), 3, cv2.LINE_AA)
    tiles.append(tile)
sc.show(hud.grid(tiles, cols=5, gap=8))
'''),
        md('''
## 1. The probe

Each degradation is applied inside the forward pass with a fixed seed, so every model sees identical images.
*Flagged* drops also tell the model which sensor is missing, which only CFFM-Net can use.
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
        md("## 2. Parameters, GFLOPs and latency\n\nGPU 0, FP16, batch 1, a 640 x 512 input, 50 warm-up and 300 "
           "timed passes with a short pause against throttling. TensorRT export is left to Phase 3."),
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
        CLEANUP,
        SAVE_NOTE,
    ])


def nb15():
    write("notebooks/phase1_still_images/15_results_and_figures.ipynb", [
        header("15 · Results, hypotheses and figures",
               "Reads every logged metric, builds the tables, decides each pilot hypothesis by its pre-registered "
               "rule and draws the figures. No number is typed by hand, so rerun this notebook whenever a run changes.",
               "`cffm-net-pilot` and the **saved outputs of notebooks 04, 05, 07, 08 and 09**",
               "a few minutes", "`results/` with tables (Markdown, LaTeX, CSV) and figures (PNG, PDF)"),
        SETUP,
        code('''
import json
from cffm import report
ev = pipeline.find_metrics(env, "eval")
pr = pipeline.find_metrics(env, "probe")
lat = {}
for root in [env.runs, *[r for r in pipeline.INPUT_ROOTS if r.exists()]]:
    for f in root.glob("**/latency/latency.json"):
        lat.update({k: v for k, v in json.loads(f.read_text()).items() if k not in lat})
print(len(ev), "evaluated runs,", len(pr), "probe results,", len(lat), "latency entries")
out = env.project / "results"; out.mkdir(exist_ok=True)
'''),
        code('''
import cv2
from cffm import showcase as sc
hero = out / "showcase" / "hero_llvip.jpg"
if hero.exists():
    sc.show(cv2.cvtColor(cv2.imread(str(hero)), cv2.COLOR_BGR2RGB))
'''),
        md("## 1. Main results\n\nAP is COCO AP50-95 on the full validation set, AP_vt / AP_t / AP_s the AI-TOD "
           "bands (<8, 8-16, 16-32 px). Cost is one T4, FP16, batch 1."),
        code('''
main = report.main_table(ev, lat)
main.round(4)
'''),
        md('''
## 2. The hypotheses, decided by the rules written down before training

Thresholds are in AP points (0.015 = 1.5 points). A difference inside the threshold is **inconclusive**, since
one seed per configuration varies by about 1 to 1.5 points.
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
        md("## 4. Tables for the write-up\n\nMarkdown for notes, CSV for spreadsheets and rule-free LaTeX for the paper."),
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

def nb16():
    write("notebooks/phase1_still_images/16_showcase.ipynb", [
        header("16 · Showcase: the Machine's view",
               "What the trained models see, drawn the way a surveillance system shows it: every subject bracketed "
               "and numbered, the thermal view beside the visible one, and a trust map of where CFFM-Net leaned on "
               "each camera. Then camera failures, a tracked street crossing, and your own images.",
               "`cffm-net-pilot`, LLVIP and M3FD, and the saved outputs of notebooks 05 and 07",
               "about 10 minutes on one GPU", "`results/showcase/`: every board as JPEG, a tracked clip as MP4 and GIF"),
        SETUP,
        code('''
from pathlib import Path

import numpy as np, torch
from cffm import hud, showcase as sc
from cffm.train import _load
from cffm.viz import make_pair

for name in ("llvip", "m3fd"):
    pipeline.prepare(name, env, plan)
ckpts = pipeline.find_checkpoints(env)
dev = "cuda:0" if torch.cuda.is_available() else "cpu"
cffm = {n: _load(ckpts[f"{n}_cffm"]).to(dev) for n in ("llvip", "m3fd")}
concat = _load(ckpts["llvip_concat"]).to(dev)
from ultralytics import YOLO
coco = YOLO(str(env.weights / "yolo26n.pt"))
out = env.project / "results" / "showcase"
out.mkdir(parents=True, exist_ok=True)
print("device:", torch.cuda.get_device_name(0) if dev != "cpu" else "cpu")
'''),
        md("## 1. One scene, three views\n\nThe most crowded night frame in LLVIP's test set. Left, the visible "
           "camera (brightened for display only); centre, thermal; right, the fusion trust map, blue where "
           "CFFM-Net weighted the visible stream and orange where it weighted thermal."),
        code('''
stem = sc.crowded(env.data / "llvip", k=1)[0]
a = hud.analyze(cffm["llvip"], *sc.pair_paths(env, "llvip", stem), frame=int(stem) % 10000)
sc.show(a.board)
sc.save(a.board, out / "hero_llvip.jpg")
sc.subject_log(a.det, a.names, a.share)
'''),
        md("## 2. Gallery\n\nThe four most crowded scenes of each test set, one per video sequence: night "
           "crossings from LLVIP and city traffic from M3FD (people, cars, buses, motorcycles, lamps, trucks)."),
        code('''
boards = []
for name in ("llvip", "m3fd"):
    for stem in sc.crowded(env.data / name, k=4):
        a = hud.analyze(cffm[name], *sc.pair_paths(env, name, stem), frame=int(stem) % 10000,
                        panels=("visible", "thermal"))
        sc.save(a.board, out / f"{name}_{stem}.jpg")
        boards.append(a.board)
sc.show(hud.grid(boards[:4], cols=2))
sc.show(hud.grid(boards[4:], cols=2))
'''),
        md("## 3. Checked against the labels\n\nColoured brackets are detections, thin white boxes the human "
           "labels. A detection counts as a match at IoU 0.5 or more."),
        code('''
from scipy.optimize import linear_sum_assignment
boards = []
for name in ("llvip", "m3fd"):
    stem = sc.crowded(env.data / name, k=6)[5]
    pair = make_pair(*sc.pair_paths(env, name, stem))
    gt = sc.gt_boxes(env, name, stem, pair.shape)
    a = hud.analyze(cffm[name], pair, gt=gt, panels=("visible",), subtitle="DETECTIONS VS LABELS")
    iou = hud._iou(a.det[:, :4], gt[:, :4]) if len(a.det) and len(gt) else np.zeros((len(a.det), len(gt)))
    r, c = linear_sum_assignment(-iou) if iou.size else ([], [])
    tp = int(sum(iou[i, j] >= 0.5 for i, j in zip(r, c)))
    print(f"{name} {stem}: {len(gt)} labels, {len(a.det)} detections, {tp} matched, "
          f"{len(gt) - tp} missed, {len(a.det) - tp} extra")
    boards.append(a.board)
sc.show(hud.grid(boards, cols=2))
'''),
        md("## 4. What the Machine trusts, by light level\n\nMean thermal share of the fusion weights against "
           "scene brightness, over 150 random test frames per dataset. If the gate works as intended, darker "
           "scenes should lean on thermal."),
        code('''
import cv2, matplotlib.pyplot as plt
from cffm.viz import reliability_maps
rng = np.random.default_rng(0)
pts = {}
for name in ("llvip", "m3fd"):
    files = sorted((env.data / name / "images" / "visible" / "val").glob("*.jpg"))
    xs, ys = [], []
    for f in rng.choice(files, size=min(150, len(files)), replace=False):
        pair = make_pair(str(f))
        share = hud.trust_share(reliability_maps(cffm[name], pair)[0], pair.shape)
        xs.append(cv2.cvtColor(pair[..., :3], cv2.COLOR_BGR2GRAY).mean())
        ys.append(float(share.mean()))
    pts[name] = (np.array(xs), np.array(ys))
plt.rcParams.update({"font.family": "serif", "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(figsize=(8, 4))
for name, col in (("llvip", "#C0622B"), ("m3fd", "#2F6DB5")):
    x, y = pts[name]
    ax.scatter(x, y, s=14, alpha=0.55, color=col, label=name.upper())
    print(f"{name}: mean thermal share {y.mean():.3f}; corr(brightness, share) = {np.corrcoef(x, y)[0, 1]:+.2f}")
ax.axhline(0.5, color="gray", lw=0.8, ls="--")
ax.set_xlabel("mean visible brightness (0-255)"); ax.set_ylabel("thermal share of fusion weight")
ax.legend(frameon=False); plt.tight_layout(); plt.savefig(out / "trust_vs_brightness.png", dpi=200); plt.show()
'''),
        md("## 5. When a camera fails\n\nThe same night scene under four conditions, CFFM-Net on the left and "
           "the concat baseline on the right. For the dropped thermal camera, CFFM-Net is told which sensor is "
           "gone through its availability flag; concat has no way to use that."),
        code('''
stem = sc.crowded(env.data / "llvip", k=2)[1]
pair = make_pair(*sc.pair_paths(env, "llvip", stem))
cases = [("clean", None, None), ("visible camera off", "visible_drop", [[0.0, 1.0]]),
         ("thermal camera off", "thermal_drop", [[1.0, 0.0]]), ("thermal misaligned 8 px", "thermal_shift", None)]
boards = []
for title, kind, flags in cases:
    p = pair if kind is None else sc.degrade(pair, kind)
    for model, label in ((cffm["llvip"], "CFFM-NET"), (concat, "CONCAT")):
        model.sensor_flags = None if flags is None or model is concat else torch.tensor(flags)
        a = hud.analyze(model, p, panels=("visible", "thermal"), title=label, subtitle=title.upper())
        model.sensor_flags = None
        boards.append(a.board)
        print(f"{title:26s} {label:9s} {len(a.det)} subjects")
sc.show(hud.grid(boards, cols=2))
_ = sc.save(hud.grid(boards, cols=2), out / "sensor_failures.jpg")
'''),
        md("## 6. Tracking a crowd\n\nLLVIP's test frames are sampled seconds apart, with different people in "
           "each, so they cannot be tracked. Tracking needs real video: MOT17-04, a pedestrian street at night "
           "filmed at 30 fps (MOT Challenge, CC BY-NC-SA 3.0). The footage is visible only, so it goes to the "
           "general YOLO26-n at 1280 px. Each subject keeps its number, a trail of where it walked and an arrow "
           "for where it is heading."),
        code('''
seq = sc.find_sequence(env, "MOT17-04")
frames = sorted(seq.glob("*.jpg"))[:300:2]
boards, n_ids = sc.track(coco, frames, title="YOLO26-N", subtitle="COCO · TRACKING · MOT17-04", panel_w=1280)
mp4 = hud.save_video(boards, out / "tracking_mot17.mp4", fps=15, width=1600)
gif = hud.save_video(boards[::3], out / "tracking_mot17.gif", fps=5, width=760)
print(f"{len(frames)} frames (10 s at 15 fps), {n_ids} track IDs; saved {Path(mp4).name}, {Path(gif).name}")
sc.show(hud.grid([boards[i] for i in (0, len(boards) // 3, 2 * len(boards) // 3, len(boards) - 1)], cols=2))
'''),
        md("## 7. Any photo\n\nCFFM-Net needs a visible and a thermal camera. For an ordinary photo we use the "
           "general YOLO26-n trained on COCO's 80 classes, drawn the same way and labelled as such."),
        code('''
from ultralytics.utils import ASSETS
boards = []
for f in sorted(ASSETS.glob("*.jpg")) + sorted((env.project / "demo" / "samples").glob("*.jpg")):
    a = hud.analyze(coco, f, title="YOLO26-N", subtitle="COCO, 80 CLASSES (NOT CFFM-NET)")
    sc.save(a.board, out / f"photo_{f.stem}.jpg")
    boards.append(a.board)
sc.show(hud.grid(boards, cols=len(boards)))
'''),
        md("## 8. Your own images\n\nSet the paths and run: a visible + thermal pair goes to CFFM-Net, a single "
           "photo to the COCO model. Running interactively, the upload box below does the same."),
        code('''
MY_VISIBLE, MY_THERMAL, MY_PHOTO = "", "", ""
if MY_VISIBLE and MY_THERMAL:
    sc.show(hud.analyze(cffm["llvip"], MY_VISIBLE, MY_THERMAL).board)
if MY_PHOTO:
    sc.show(hud.analyze(coco, MY_PHOTO, title="YOLO26-N", subtitle="COCO, 80 CLASSES").board)
try:
    import ipywidgets as w
    from IPython.display import display
    up = w.FileUpload(accept="image/*", multiple=True, description="upload")
    def on_upload(change):
        files = [np.frombuffer(bytes(f["content"]), np.uint8) for f in up.value]
        ims = [cv2.imdecode(b, cv2.IMREAD_COLOR) for b in files]
        a = hud.analyze(cffm["llvip"], ims[0], cv2.cvtColor(ims[1], cv2.COLOR_BGR2GRAY)) if len(ims) == 2 else \\
            hud.analyze(coco, ims[0], title="YOLO26-N", subtitle="COCO, 80 CLASSES")
        sc.show(a.board)
    up.observe(on_upload, names="value")
    display(up)
except ImportError:
    print("ipywidgets not installed; use the paths above")
'''),
    ])


if __name__ == "__main__":
    import sys
    every = (nb00, nb01, nb02, nb03, nb04, nb05, nb06, nb07, nb08, nb08b, nb08c, nb08d, nb08e, nb08f, nb09, nb15, nb16)
    pick = set(sys.argv[1:])
    for f in every:
        if not pick or f.__name__[2:] in pick:
            f()
