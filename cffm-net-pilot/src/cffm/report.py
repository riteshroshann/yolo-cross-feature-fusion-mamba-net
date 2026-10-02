"""
Turning logged metrics into tables and verdicts.

Kept out of the notebooks on purpose: the decision rules for the pilot
hypotheses were fixed before any training (pilot note, Table 4), and code that
decides whether a hypothesis holds deserves unit tests, not a notebook cell.
"""
from __future__ import annotations

import math

import pandas as pd

AP = "mAP50-95(B)"
MAIN_COLS = {"mAP50-95(B)": "AP", "mAP50(B)": "AP50", "mAP_small(B)": "AP_small",
             "AP_vt(B)": "AP_vt", "AP_t(B)": "AP_t", "AP_s(B)": "AP_s"}
ORDER = ["llvip_yolo26n_visible", "llvip_yolo26n_thermal", "llvip_concat", "llvip_cffm", "llvip_cffm_nogate",
         "llvip_cffm_gconv", "llvip_cffm_nop2", "m3fd_concat", "m3fd_cffm"]


def _get(d: dict, run: str, key: str = AP) -> float:
    return float(d.get(run, {}).get(key, math.nan))


def main_table(ev: dict, lat: dict, order=ORDER) -> pd.DataFrame:
    """One row per run: accuracy (full validation set) and cost (one T4, FP16, batch 1)."""
    rows = []
    for run in [r for r in order if r in ev] + sorted(set(ev) - set(order)):
        row = {"run": run, **{v: _get(ev, run, k) for k, v in MAIN_COLS.items()}}
        row.update({k: float(lat.get(run, {}).get(k, math.nan)) for k in ("params_M", "GFLOPs", "median_ms")})
        rows.append(row)
    return pd.DataFrame(rows).set_index("run") if rows else pd.DataFrame()


def probe_table(pr: dict, runs, kinds, flagged=()) -> pd.DataFrame:
    """AP under each test-time degradation: rows = probe, columns = run. Exact key matching only."""
    rows = []
    for run in runs:
        for kind in kinds:
            for fl in ([False, True] if kind in flagged else [False]):
                key = f"{run}_{kind}" + ("_flagged" if fl else "")
                if key in pr:
                    rows.append({"run": run, "probe": kind + (" (flagged)" if fl else ""), "AP": _get(pr, key)})
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    order = list(dict.fromkeys(df["probe"]))
    return df.pivot(index="probe", columns="run", values="AP").loc[order]


def verdict(diff: float, threshold: float) -> str:
    if math.isnan(diff):
        return "missing run"
    if diff >= threshold:
        return "supported"
    return "against" if diff <= -threshold else "inconclusive"


def mean_degradation_advantage(pr: dict, model: str, reference: str, kinds) -> float:
    """How much less `model` loses than `reference` under degradation, averaged over probes.

    loss(x) = AP(x, clean) - AP(x, probe); advantage = loss(reference) - loss(model).
    Positive means `model` degrades less.
    """
    vals = []
    for k in kinds:
        if k == "clean":
            continue
        need = [f"{model}_clean", f"{model}_{k}", f"{reference}_clean", f"{reference}_{k}"]
        if all(n in pr for n in need):
            m_loss = _get(pr, need[0]) - _get(pr, need[1])
            r_loss = _get(pr, need[2]) - _get(pr, need[3])
            vals.append(r_loss - m_loss)
    return sum(vals) / len(vals) if vals else math.nan


def decide(ev: dict, pr: dict, lat: dict, kinds) -> pd.DataFrame:
    """The five pilot hypotheses, decided by the rules written down before training."""
    H = []
    d = _get(ev, "llvip_cffm") - _get(ev, "llvip_yolo26n_thermal")
    H.append(("PH1", "fusion beats the best single sensor at night", f"{d:+.4f} AP vs thermal-only",
              verdict(d, 0.015)))

    d_clean = _get(ev, "llvip_cffm") - _get(ev, "llvip_cffm_nogate")
    d_probe = mean_degradation_advantage(pr, "llvip_cffm", "llvip_cffm_nogate", kinds)
    best = d_clean if math.isnan(d_probe) else max(d_clean, d_probe)
    H.append(("PH2", "gating the step adds value",
              f"clean {d_clean:+.4f} AP; degrades less by {d_probe:+.4f} AP", verdict(best, 0.015)))

    d = sum(_get(ev, "m3fd_cffm", k) - _get(ev, "m3fd_concat", k) for k in ("AP_t(B)", "AP_s(B)")) / 2
    H.append(("PH3", "the pathway helps 8-32 px objects (M3FD)", f"{d:+.4f} AP in the 8-32 px bands",
              verdict(d, 0.015)))

    d = _get(ev, "llvip_cffm") - _get(ev, "llvip_cffm_gconv")
    H.append(("PH4", "the selective scan is needed", f"{d:+.4f} AP vs gated-conv control", verdict(d, 0.010)))

    num = float(lat.get("llvip_cffm", {}).get("median_ms", math.nan))
    den = float(lat.get("llvip_concat", {}).get("median_ms", math.nan))
    r = num / den if den == den and den else math.nan
    H.append(("PH5", "latency at most 2x the two-stream baseline", f"ratio {r:.2f}",
              "missing run" if math.isnan(r) else ("supported" if r <= 2 else "against")))
    return pd.DataFrame(H, columns=["id", "hypothesis", "evidence", "verdict"]).set_index("id")


# --------------------------------------------------------------------------- #
# writers (no extra dependencies: no tabulate, no jinja2)
# --------------------------------------------------------------------------- #
def _fmt(v, digits):
    if isinstance(v, float):
        return "" if math.isnan(v) else f"{v:.{digits}f}"
    return str(v)


def to_markdown(df: pd.DataFrame, digits: int = 4) -> str:
    cols = [df.index.name or ""] + [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for i, row in zip(df.index, df.values):
        lines.append("| " + " | ".join([str(i)] + [_fmt(v, digits) for v in row]) + " |")
    return "\n".join(lines) + "\n"


def to_latex(df: pd.DataFrame, caption: str, label: str, digits: int = 3) -> str:
    """A table in the dossier's style: no rules, spacing only."""
    esc = lambda s: str(s).replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")
    align = "l" + "".join("r" if pd.api.types.is_numeric_dtype(df[c]) else "l" for c in df.columns)
    head = " & ".join([esc(df.index.name or "")] + [esc(c) for c in df.columns]) + r" \\ \addlinespace[5pt]"
    body = "\n".join(" & ".join([esc(i)] + [esc(_fmt(v, digits)) for v in row]) + r" \\"
                     for i, row in zip(df.index, df.values))
    return ("\\begin{table}[htbp]\n\\centering\n"
            f"\\caption{{{caption}}}\n\\label{{{label}}}\n\\small\n"
            f"\\begin{{tabular}}{{@{{}}{align}@{{}}}}\n{head}\n{body}\n\\end{{tabular}}\n\\end{{table}}\n")
