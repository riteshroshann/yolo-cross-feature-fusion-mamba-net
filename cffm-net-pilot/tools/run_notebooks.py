"""Execute notebooks in place on this machine and clean their outputs for GitHub.

    python tools/run_notebooks.py 03 07 16
"""
import json
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EMOJI = re.compile("[\U0001F000-\U0001FAFF\u2600-\u27BF\u2B00-\u2BFF\uFE0F\u200D]")
ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
KEEP_HEAD, KEEP_TAIL = 25, 80


def clean_text(t):
    t = EMOJI.sub("", ANSI.sub("", t))
    lines = [l.split("\r")[-1] for l in t.split("\n")]
    if len(lines) > KEEP_HEAD + KEEP_TAIL + 10:
        n = len(lines) - KEEP_HEAD - KEEP_TAIL
        lines = lines[:KEEP_HEAD] + [f"[{n} log lines trimmed]"] + lines[-KEEP_TAIL:]
    return "\n".join(lines)


def clean(nb_path):
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    for c in nb["cells"]:
        for o in c.get("outputs", []):
            if "text" in o:
                o["text"] = clean_text("".join(o["text"]))
            for k in ("text/plain", "text/html", "text/markdown"):
                if k in o.get("data", {}):
                    o["data"][k] = EMOJI.sub("", ANSI.sub("", "".join(o["data"][k])))
            if o.get("output_type") == "error":
                o["traceback"] = [ANSI.sub("", l) for l in o["traceback"]]
    nb_path.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


def run(nb_path, timeout=6 * 3600):
    jupyter = Path(sys.executable).with_name("jupyter.exe" if sys.platform == "win32" else "jupyter")
    t = time.time()
    r = subprocess.run([str(jupyter), "nbconvert", "--to", "notebook", "--execute", "--inplace",
                        f"--ExecutePreprocessor.timeout={timeout}", "--ExecutePreprocessor.kernel_name=python3",
                        nb_path.name], cwd=nb_path.parent, capture_output=True, text=True, encoding="utf-8",
                       errors="replace")
    clean(nb_path)
    ok = r.returncode == 0
    print(f"{nb_path.name:44s} {'ok' if ok else 'FAILED'}  {(time.time() - t) / 60:5.1f} min", flush=True)
    if not ok:
        print("\n".join(r.stderr.strip().splitlines()[-25:]), flush=True)
    return ok


if __name__ == "__main__":
    wanted = sys.argv[1:]
    books = sorted((ROOT / "notebooks").glob("*/*.ipynb"))
    for nb in books:
        if not wanted or nb.name.split("_")[0] in wanted:
            run(nb)
