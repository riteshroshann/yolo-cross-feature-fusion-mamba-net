"""Compile every fig_*.tex here to PDF (Tectonic) and render a 300-dpi PNG next to it.

    TECTONIC=/path/to/tectonic python build_figures.py
"""
import os, subprocess, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
tectonic = os.environ.get("TECTONIC", "tectonic")
for tex in sorted(HERE.glob("fig_*.tex")):
    r = subprocess.run([tectonic, "-X", "compile", tex.name], cwd=HERE, capture_output=True, text=True)
    if r.returncode:
        print(tex.name, "FAILED\n", "\n".join(l for l in r.stderr.splitlines() if "error" in l.lower())[:2000])
        continue
    import pymupdf
    doc = pymupdf.open(HERE / tex.with_suffix(".pdf").name)
    doc[0].get_pixmap(dpi=300).save(str(HERE / tex.with_suffix(".png").name))
    print("built", tex.stem)
