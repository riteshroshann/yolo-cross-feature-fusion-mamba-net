"""Write refs.tex with every cited work, from extra_refs.tex and the dossier bibliography, in citation order."""
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCES = [HERE / "extra_refs.tex", HERE.parents[1] / "study-prep" / "guide" / "extra_refs.tex", HERE.parents[1] / "docs" / "dossier" / "references.tex"]

text = "".join(p.read_text(encoding="utf-8") for p in sorted((HERE / "chapters").glob("*.tex")))
cited = list(dict.fromkeys(k.strip() for m in re.finditer(r"\\cite[pt]?\{([^}]+)\}", text) for k in m.group(1).split(",")))
items = {}
for src in SOURCES:
    body = src.read_text(encoding="utf-8")
    for it in re.split(r"(?=\\bibitem\{)", body)[1:]:
        key = re.match(r"\\bibitem\{([^}]+)\}", it).group(1)
        items.setdefault(key, it.split("\\end{thebibliography}")[0].rstrip())
missing = [k for k in cited if k not in items]
if missing:
    raise SystemExit(f"cited but not found: {missing}")
(HERE / "refs.tex").write_text("\\begin{thebibliography}{99}\n\n" + "\n\n".join(items[k] for k in cited) +
                               "\n\n\\end{thebibliography}\n", encoding="utf-8")
print(f"refs.tex: {len(cited)} entries")
