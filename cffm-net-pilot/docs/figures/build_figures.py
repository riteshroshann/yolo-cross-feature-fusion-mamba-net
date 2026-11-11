"""Compile fig_*.tex here (or the names given) to PDF with Tectonic and render a 4K PNG (3840 px on the long edge) next to each.

    TECTONIC=/path/to/tectonic python build_figures.py [fig_name ...]
"""
import os
import subprocess
import sys
from pathlib import Path

import pymupdf

HERE = Path(__file__).resolve().parent
tectonic = os.environ.get("TECTONIC", "tectonic")
pick = {a.removesuffix(".tex") for a in sys.argv[1:]}
for tex in sorted(HERE.glob("fig_*.tex")):
    if pick and tex.stem not in pick:
        continue
    r = subprocess.run([tectonic, "-X", "compile", tex.name], cwd=HERE, capture_output=True, text=True)
    if r.returncode:
        print(tex.name, "FAILED\n", "\n".join(l for l in r.stderr.splitlines() if "error" in l.lower())[:2000])
        continue
    doc = pymupdf.open(HERE / tex.with_suffix(".pdf").name)
    page = doc[0]
    zoom = 3840 / max(page.rect.width, page.rect.height)
    page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom)).save(str(HERE / tex.with_suffix(".png").name))
    print("built", tex.stem, f"{doc[0].rect.width / 72 * 2.54:.1f} x {doc[0].rect.height / 72 * 2.54:.1f} cm")
