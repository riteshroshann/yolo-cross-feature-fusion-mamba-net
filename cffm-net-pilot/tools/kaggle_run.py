"""Run the pilot notebooks on Kaggle through its CLI (kaggle>=1.8, token in ~/.kaggle/access_token).

    python tools/kaggle_run.py upload | push 00 | status [00 02 ...] | logs 00 | wait 04 05
"""
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT.parent / "dist" / "kaggle"
KAGGLE = shutil.which("kaggle") or str(Path(sys.executable).parent / "kaggle")

LLVIP = "afradhossain/llvip-dataset"
M3FD = "nus1998/m3fd-dataset"
MOT17 = "ahmedsamir1598/mot17challenge"
P0, P1 = "notebooks/phase0_environment", "notebooks/phase1_still_images"

NOTEBOOKS = {
    "00": (f"{P0}/00_environment_setup.ipynb", "gpu", [], []),
    "01": (f"{P0}/01_data_acquisition.ipynb", "cpu", [LLVIP, M3FD], []),
    "02": (f"{P0}/02_data_conversion.ipynb", "cpu", [LLVIP, M3FD], []),
    "03": (f"{P1}/03_data_audit.ipynb", "cpu", [LLVIP, M3FD], ["02"]),
    "04": (f"{P1}/04_baselines_single_modality.ipynb", "gpu", [LLVIP], ["02"]),
    "05": (f"{P1}/05_baseline_two_stream.ipynb", "gpu", [LLVIP, M3FD], ["02"]),
    "06": (f"{P1}/06_inside_cmfm.ipynb", "cpu", [], []),
    "07": (f"{P1}/07_train_cffm_net.ipynb", "gpu", [LLVIP, M3FD], ["02"]),
    "08": (f"{P1}/08_ablations.ipynb", "gpu", [LLVIP], ["02"]),
    "08b": (f"{P1}/08b_head_control.ipynb", "gpu", [LLVIP], ["02"]),
    "08c": (f"{P1}/08c_round2_llvip.ipynb", "gpu", [LLVIP], ["02"]),
    "08d": (f"{P1}/08d_round2_m3fd.ipynb", "gpu", [M3FD], ["02"]),
    "09": (f"{P1}/09_probe_and_latency.ipynb", "gpu", [LLVIP], ["02", "04", "05", "07", "08"]),
    "15": (f"{P1}/15_results_and_figures.ipynb", "cpu", [], ["04", "05", "07", "08", "09"]),
    "16": (f"{P1}/16_showcase.ipynb", "gpu", [LLVIP, M3FD, MOT17], ["05", "07"]),
}


def kaggle(*args, check=True) -> str:
    r = subprocess.run([KAGGLE, *args], capture_output=True, text=True, encoding="utf-8", errors="replace")
    if check and r.returncode:
        raise SystemExit(f"kaggle {' '.join(args)} failed:\n{r.stdout}\n{r.stderr}")
    return r.stdout + r.stderr


def user() -> str:
    m = re.search(r"username:\s*(\S+)", kaggle("config", "view"))
    if not m:
        raise SystemExit("no Kaggle username configured; is the API token in ~/.kaggle/access_token?")
    return m.group(1)


def slug(nb_id: str) -> str:
    return "cffm-" + Path(NOTEBOOKS[nb_id][0]).stem.replace("_", "-")


def upload(message="update"):
    zip_path = ROOT.parent / "dist" / "cffm-net-pilot.zip"
    if not zip_path.exists():
        raise SystemExit("build the zip first: python tools/make_zip.py")
    d = STAGE / "dataset"
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
    shutil.copy(zip_path, d / zip_path.name)
    ref = f"{user()}/cffm-net-pilot"
    (d / "dataset-metadata.json").write_text(json.dumps(
        {"title": "cffm-net-pilot", "id": ref, "licenses": [{"name": "other"}]}, indent=2))
    exists = "cffm-net-pilot" in kaggle("datasets", "list", "--mine", check=False)
    print(kaggle("datasets", "version", "-p", str(d), "-m", message) if exists else
          kaggle("datasets", "create", "-p", str(d)))
    for _ in range(60):
        if "ready" in kaggle("datasets", "status", ref, check=False):
            print(ref, "ready")
            return
        time.sleep(10)
    print(ref, "still processing")


def push(nb_id: str):
    path, machine, datasets, needs = NOTEBOOKS[nb_id]
    u = user()
    d = STAGE / slug(nb_id)
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
    shutil.copy(ROOT / path, d / Path(path).name)
    meta = {
        "id": f"{u}/{slug(nb_id)}", "title": slug(nb_id), "code_file": Path(path).name,
        "language": "python", "kernel_type": "notebook", "is_private": True,
        "enable_gpu": machine == "gpu", "enable_tpu": False, "enable_internet": True,
        "machine_shape": "NvidiaTeslaT4" if machine == "gpu" else "",
        "dataset_sources": [f"{u}/cffm-net-pilot", *datasets],
        "kernel_sources": [f"{u}/{slug(n)}" for n in needs],
        "competition_sources": [], "model_sources": [],
    }
    (d / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
    print(kaggle("kernels", "push", "-p", str(d)))


def status(nb_id: str) -> str:
    out = kaggle("kernels", "status", f"{user()}/{slug(nb_id)}", check=False)
    m = re.search(r'status "?(?:KernelWorkerStatus\.)?(\w+)', out)
    return m.group(1).lower() if m else out.strip()


def wait(ids, every=120):
    pending = list(ids)
    while pending:
        for i in list(pending):
            s = status(i)
            if s not in ("queued", "running"):
                print(f"{i} {slug(i)}: {s}", flush=True)
                pending.remove(i)
        if pending:
            time.sleep(every)


if __name__ == "__main__":
    sys.stdout.reconfigure(errors="replace")
    cmd, *ids = sys.argv[1:] or ["status"]
    if cmd == "upload":
        upload(" ".join(ids) or "update")
    elif cmd == "push":
        for i in ids:
            push(i)
    elif cmd == "status":
        for i in ids or NOTEBOOKS:
            print(f"{i} {slug(i):40s} {status(i)}")
    elif cmd == "logs":
        print(kaggle("kernels", "logs", f"{user()}/{slug(ids[0])}", check=False))
    elif cmd == "wait":
        wait(ids)
    else:
        raise SystemExit(__doc__)
