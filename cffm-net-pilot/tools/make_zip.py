"""
Pack the project into cffm-net-pilot.zip, the file you upload to Kaggle as a dataset.

    python tools/make_zip.py            # writes ../dist/cffm-net-pilot.zip

Everything inside the zip sits under one top folder, cffm-net-pilot/, so the notebooks find it at
/kaggle/input/<dataset>/cffm-net-pilot/. Anything generated (data, runs, caches, build files) is left out,
and the novelty PDF from ../docs/novelty is added under docs/ so the zip is self-contained.
"""
import fnmatch
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT.parent / "dist" / "cffm-net-pilot.zip"

SKIP_DIRS = {"data", "runs", "results", "__pycache__", ".pytest_cache", ".ipynb_checkpoints", ".git", ".venv"}
SKIP_FILES = ["*.pyc", "*.egg-info", "*.aux", "*.log", "*.zip"]
EXTRA = {ROOT.parent / "docs" / "novelty" / "CFFM-Net_Pilot_Novelty.pdf": "docs/CFFM-Net_Pilot_Novelty.pdf"}


def wanted(p: Path) -> bool:
    rel = p.relative_to(ROOT)
    if any(part in SKIP_DIRS or part.endswith(".egg-info") for part in rel.parts[:-1]):
        return False
    return not any(fnmatch.fnmatch(p.name, pat) for pat in SKIP_FILES)


def main():
    OUT.parent.mkdir(exist_ok=True)
    files = sorted(p for p in ROOT.rglob("*") if p.is_file() and wanted(p))
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for p in files:
            z.write(p, f"cffm-net-pilot/{p.relative_to(ROOT).as_posix()}")
        for src, dst in EXTRA.items():
            if src.exists():
                z.write(src, f"cffm-net-pilot/{dst}")
            else:
                print("note: not found, skipped:", src)
    print(f"{OUT}  {OUT.stat().st_size / 1e6:.1f} MB, {len(files) + sum(s.exists() for s in EXTRA)} files")


if __name__ == "__main__":
    main()
