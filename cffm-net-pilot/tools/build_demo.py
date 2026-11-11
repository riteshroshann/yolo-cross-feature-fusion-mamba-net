"""Fill demo/ with sample pairs, sample photos and weights, so the app runs anywhere (locally, Hugging Face).

    python tools/build_demo.py            # needs data/processed and runs/ from the pilot
    python tools/build_demo.py --space    # also stage dist/hf-space/ ready to push to a Hugging Face Space
    python tools/build_demo.py --boards   # also render each sample to web/assets/boards/ for the static site
"""
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cffm import env as cenv, showcase as sc

DEMO = ROOT / "demo"
RUNS = {"llvip_cffm": "llvip_cffm", "m3fd_cffm": "m3fd_cffm"}


def _save_board(a, out: Path, dev: str):
    import cv2
    cv2.imwrite(str(out.with_suffix(".jpg")), cv2.cvtColor(a.board, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 88])
    df = sc.subject_log(a.det, a.names, a.share)
    df = df.reset_index() if len(df) else df
    log = {"headers": [str(c) for c in df.columns], "data": json.loads(df.to_json(orient="values"))}
    msg = f"{len(a.det)} subjects · {a.ms:.0f} ms on {dev}"
    out.with_suffix(".json").write_text(json.dumps({"msg": msg, "log": log}), encoding="utf-8")


def build_boards():
    """Render every demo sample once so the site can show results without a backend."""
    import cv2
    import torch
    from ultralytics import YOLO

    from cffm import hud
    from cffm.train import _load

    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    name = torch.cuda.get_device_name(0) if dev != "cpu" else "CPU"
    out = ROOT / "web" / "assets" / "boards"
    out.mkdir(parents=True, exist_ok=True)
    models = {k: _load(DEMO / "weights" / f"{k}_cffm.pt").to(dev) for k in ("llvip", "m3fd")}
    for v in sorted((DEMO / "samples" / "pairs").glob("*_visible.*")):
        t = v.with_name(v.name.replace("_visible", "_thermal"))
        model = models[v.name.split("_")[0]]
        for _ in range(2):
            a = hud.analyze(model, cv2.imread(str(v)), cv2.imread(str(t), cv2.IMREAD_GRAYSCALE), conf=0.3)
        _save_board(a, out / v.stem.replace("_visible", ""), name)
    coco = YOLO(DEMO / "weights" / "yolo26n.pt")
    for f in sorted((DEMO / "samples" / "photos").glob("*.jpg")):
        for _ in range(2):
            a = hud.analyze(coco, cv2.imread(str(f)), conf=0.3, title="YOLO26-N", subtitle="COCO, 80 CLASSES (NOT CFFM-NET)")
        _save_board(a, out / f"photo_{f.stem}", name)
    print("boards:", len(list(out.glob("*.jpg"))))


def main():
    env = cenv.setup(verbose=False)
    pairs, photos, w = DEMO / "samples" / "pairs", DEMO / "samples" / "photos", DEMO / "weights"
    for d in (pairs, photos, w):
        d.mkdir(parents=True, exist_ok=True)
    for name, k in (("llvip", 4), ("m3fd", 3)):
        for stem in sc.crowded(env.data / name, k=k):
            v, t = sc.pair_paths(env, name, stem)
            shutil.copy(v, pairs / f"{name}_{stem}_visible{v.suffix}")
            shutil.copy(t, pairs / f"{name}_{stem}_thermal{t.suffix}")
    from ultralytics.utils import ASSETS
    for f in ASSETS.glob("*.jpg"):
        shutil.copy(f, photos / f.name)
    for run, out in RUNS.items():
        shutil.copy(ROOT / "runs" / run / "weights" / "best.pt", w / f"{out}.pt")
    shutil.copy(ROOT / "weights" / "yolo26n.pt", w / "yolo26n.pt")
    print("demo/ ready:", sum(1 for _ in pairs.iterdir()) // 2, "pairs,", sum(1 for _ in photos.iterdir()), "photos")
    if "--space" in sys.argv:
        space = ROOT.parent / "dist" / "hf-space"
        shutil.rmtree(space, ignore_errors=True)
        shutil.copytree(DEMO, space, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "flagged"))
        shutil.copytree(ROOT / "src" / "cffm", space / "cffm", ignore=shutil.ignore_patterns("__pycache__"))
        print("staged", space)
    if "--boards" in sys.argv:
        build_boards()


if __name__ == "__main__":
    main()
