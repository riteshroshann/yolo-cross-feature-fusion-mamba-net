"""Fill demo/ with sample pairs, sample photos and weights, so the app runs anywhere (locally, Hugging Face).

    python tools/build_demo.py            # needs data/processed and runs/ from the pilot
    python tools/build_demo.py --space    # also stage dist/hf-space/ ready to push to a Hugging Face Space
"""
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cffm import env as cenv, showcase as sc

DEMO = ROOT / "demo"
RUNS = {"llvip_cffm": "llvip_cffm", "m3fd_cffm": "m3fd_cffm"}


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


if __name__ == "__main__":
    main()
