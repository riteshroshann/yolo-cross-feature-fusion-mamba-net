"""Build refs.tex for the novelty note from the dossier bibliography.

Only the entries actually cited in novelty.tex are kept, in the dossier's
order, so the two documents never disagree about a reference.
"""
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
BIB = os.path.join(HERE, "..", "dossier", "references.tex")

text = open(os.path.join(HERE, "novelty.tex"), encoding="utf-8").read()
cited = {k.strip() for m in re.finditer(r"\\cite[pt]?\{([^}]+)\}", text) for k in m.group(1).split(",")}

bib = open(BIB, encoding="utf-8").read()
body = bib[bib.index(r"\begin{thebibliography}"):bib.index(r"\end{thebibliography}")]
items = re.split(r"(?=\\bibitem\{)", body)[1:]
keep = [it.rstrip() for it in items if re.match(r"\\bibitem\{([^}]+)\}", it).group(1) in cited]

found = {re.match(r"\\bibitem\{([^}]+)\}", it).group(1) for it in keep}
missing = sorted(cited - found)
if missing:
    raise SystemExit(f"cited but not in the dossier bibliography: {missing}")

with open(os.path.join(HERE, "refs.tex"), "w", encoding="utf-8") as f:
    f.write("\\begin{thebibliography}{99}\n\n" + "\n\n".join(keep) + "\n\n\\end{thebibliography}\n")
print(f"refs.tex: {len(keep)} entries")
