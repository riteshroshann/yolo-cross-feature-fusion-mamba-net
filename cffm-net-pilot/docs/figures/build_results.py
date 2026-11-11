"""Regenerate every res_*.pdf and res_*.png result figure from the stored result files."""
import csv
import io
import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PILOT = ROOT / "cffm-net-pilot"
RES = PILOT / "results"
RUNS = PILOT / "runs"
DIST = ROOT / "dist" / "out"

WIDTH = 7.16
PIXELS = 3840
SEEDS = (0, 1, 2)
SEED_DIR = {("llvip", 1): "08g", ("llvip", 2): "08h", ("m3fd", 1): "08i", ("m3fd", 2): "08j"}
V3_DIR = {"llvip": "08k-v3-llvip", "m3fd": "08l-v3-m3fd"}
DS_NAME = {"llvip": "LLVIP", "m3fd": "M3FD"}

STYLE = {
    "v2": {"label": "CFFM-Net v2", "color": "#3B8B66", "marker": "o"},
    "md": {"label": "Concat (modality dropout)", "color": "#5E6C7C", "marker": "s"},
    "ft": {"label": "Concat (fault-trained)", "color": "#A3ACB6", "marker": "^"},
    "v3": {"label": "CFFM-Net v3", "color": "#7450A8", "marker": "D"},
    "thermal": {"label": "Thermal only", "color": "#C0622B", "marker": "v"},
    "visible": {"label": "Visible only", "color": "#2F6DB5", "marker": "P"},
}
INK = "#1E2227"
MUTED = "#5C6670"
OFF_COLOR = "#A3ACB6"
ON_COLOR = "#7450A8"

FAULTS = ["visible_dark", "visible_glare", "visible_fog", "visible_noise", "visible_blur", "visible_drop",
          "thermal_drop", "thermal_shift", "thermal_noise", "thermal_crossover", "thermal_dead"]
FAULT_LABEL = {
    "visible_dark": "Visible dark", "visible_glare": "Visible glare", "visible_fog": "Visible fog",
    "visible_noise": "Visible noise", "visible_blur": "Visible blur", "visible_drop": "Visible drop",
    "thermal_drop": "Thermal drop", "thermal_shift": "Thermal shift", "thermal_noise": "Thermal noise",
    "thermal_crossover": "Thermal crossover", "thermal_dead": "Thermal dead",
}
VISIBLE_FAULTS = 6


def setup_style() -> None:
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["STIXGeneral"], "mathtext.fontset": "stix",
        "font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7, "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5, "legend.fontsize": 6.5, "axes.linewidth": 0.5,
        "axes.spines.top": False, "axes.spines.right": False, "xtick.major.width": 0.5,
        "ytick.major.width": 0.5, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "lines.linewidth": 1.0, "axes.edgecolor": INK, "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": INK, "ytick.color": INK, "savefig.facecolor": "white", "axes.unicode_minus": True,
    })


def save(fig: plt.Figure, name: str) -> None:
    fig.savefig(HERE / f"res_{name}.pdf")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=PIXELS / WIDTH)
    img = Image.open(buf)
    if img.width != PIXELS:
        img = img.resize((PIXELS, round(img.height * PIXELS / img.width)), Image.LANCZOS)
    img.save(HERE / f"res_{name}.png")
    plt.close(fig)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def metrics_path(ds: str, model: str, seed: int) -> Path:
    if model in ("visible", "thermal"):
        suffix = "_b32" if ds == "m3fd" else ""
        return RUNS / "eval" / f"{ds}_yolo26n_{model}{suffix}" / "metrics.json"
    if model in ("v3", "ft"):
        name = "cffm_v3_e80" if model == "v3" else "concat_ft_e80"
        return DIST / V3_DIR[ds] / "runs" / "eval" / f"{ds}_{name}" / "metrics.json"
    name = "cffm_v2_e80" if model == "v2" else "concat_md_e80"
    if seed == 0:
        return RUNS / "faults" / f"{ds}_{name}" / "clean" / "metrics.json"
    return DIST / SEED_DIR[(ds, seed)] / "runs" / "eval" / f"{ds}_{name}_s{seed}" / "metrics.json"


def run_metric(ds: str, model: str, seed: int, key: str = "mAP50-95(B)") -> float:
    return 100 * read_json(metrics_path(ds, model, seed))[key]


def seed_values(ds: str, model: str, key: str = "mAP50-95(B)") -> list[float]:
    seeds = SEEDS if model in ("v2", "md") else (0,)
    return [run_metric(ds, model, s, key) for s in seeds]


def mean_std(values: list[float]) -> tuple[float, float]:
    return statistics.fmean(values), (statistics.stdev(values) if len(values) > 1 else 0.0)


def load_faults() -> dict[tuple[str, str, str, int], list[float]]:
    """Map (dataset, model, fault, severity) to mAP50-95 in percent, ordered by seed."""
    kinds = {"cffm_v2": "v2", "concat_md": "md", "concat_ft": "ft"}
    table: dict[tuple[str, str, str, int, int], float] = {}
    with open(RES / "faults.csv", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["flagged"] == "True":
                continue
            run = row["run"]
            seed = int(run.rsplit("_s", 1)[1]) if run[-3:-1] == "_s" else 0
            model = next(v for k, v in kinds.items() if k in run)
            table[(row["dataset"], model, row["fault"], int(row["severity"]), seed)] = 100 * float(row["AP"])
    out: dict[tuple[str, str, str, int], list[float]] = {}
    for (ds, model, fault, sev, seed), v in sorted(table.items()):
        out.setdefault((ds, model, fault, sev), []).append(v)
    return out


def fault_mean(table: dict, ds: str, model: str, fault: str, sev: int) -> float:
    return statistics.fmean(table[(ds, model, fault, sev)])


def luminance_text(cmap, norm, value: float) -> str:
    r, g, b, _ = cmap(norm(value))
    return "black" if 0.2126 * r + 0.7152 * g + 0.0722 * b > 0.5 else "white"


def draw_heat(ax, matrix: np.ndarray, cmap, norm, fmt: str, show_rows: bool, cell_size: float = 6.0):
    im = ax.imshow(matrix, cmap=cmap, norm=norm, aspect="auto", interpolation="nearest")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, fmt.format(matrix[i, j] + 0.0).replace("-0.0", "0.0").replace("-", "−").replace("+0.0", "0.0"), ha="center", va="center",
                    fontsize=cell_size, color=luminance_text(cmap, norm, matrix[i, j]))
    ax.set_xticks(range(matrix.shape[1]), ["1", "2", "3"])
    ax.set_yticks(range(matrix.shape[0]), [FAULT_LABEL[f] for f in FAULTS] if show_rows else [])
    ax.tick_params(length=0)
    ax.axhline(VISIBLE_FAULTS - 0.5, color="white", lw=1.6)
    for spine in ax.spines.values():
        spine.set_visible(False)
    return im


def fig_main() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.7), layout="constrained")
    ranges = {"llvip": (50, 70), "m3fd": (46, 56)}
    for ax, ds, tag in zip(axes, ("llvip", "m3fd"), "ab"):
        for x, model in enumerate(("v2", "md")):
            vals = seed_values(ds, model)
            mean, std = mean_std(vals)
            s = STYLE[model]
            jitter = np.linspace(-0.12, 0.12, len(vals))
            ax.scatter(x + jitter, vals, s=16, color=s["color"], marker=s["marker"], zorder=3,
                       edgecolor="white", linewidth=0.4)
            ax.errorbar(x + 0.3, mean, yerr=std, fmt="_", color=INK, capsize=3, markersize=9, lw=0.8, zorder=4)
            ax.text(x + 0.38, mean, f"{mean:.1f} ± {std:.1f}", va="center", fontsize=6.5)
        for model, dash in (("visible", (0, (4, 2))), ("thermal", (0, (1, 1.5)))):
            v = run_metric(ds, model, 0)
            ax.axhline(v, color=STYLE[model]["color"], lw=0.9, ls=dash, zorder=1)
            ax.text(-0.46, v, f"{STYLE[model]['label']} {v:.1f}", ha="left", va="bottom",
                    fontsize=6.5, color=STYLE[model]["color"])
        ax.set_xlim(-0.5, 1.9)
        ax.set_ylim(*ranges[ds])
        ax.set_xticks([0, 1], [STYLE["v2"]["label"], "Concat\n(modality dropout)"])
        ax.set_ylabel(f"mAP50–95 (%), axis starts at {ranges[ds][0]}")
        note = "single cameras: 30 epochs" if ds == "llvip" else "single cameras: 80 epochs"
        ax.set_title(f"({tag}) {DS_NAME[ds]}", loc="left")
        ax.text(0.99, 0.02, note, transform=ax.transAxes, ha="right", va="bottom", fontsize=6, color=MUTED)
    save(fig, "main")


def fig_fault_heatmap() -> None:
    table = load_faults()
    models = ("v2", "md", "ft")
    mats = {m: np.array([[fault_mean(table, "m3fd", m, f, s) for s in (1, 2, 3)] for f in FAULTS]) for m in models}
    lo = min(m.min() for m in mats.values())
    hi = max(m.max() for m in mats.values())
    norm = Normalize(vmin=np.floor(lo / 5) * 5, vmax=np.ceil(hi / 5) * 5)
    cmap = plt.get_cmap("viridis")
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH, 3.7), layout="constrained")
    heads = {"v2": "three-seed mean", "md": "three-seed mean", "ft": "seed 0"}
    for ax, m, tag in zip(axes, models, "abc"):
        im = draw_heat(ax, mats[m], cmap, norm, "{:.1f}", m == "v2")
        ax.set_title(f"({tag}) {STYLE[m]['label']}\n{heads[m]}", loc="left")
        ax.set_xlabel("Fault severity")
    cbar = fig.colorbar(im, ax=axes, shrink=0.8, pad=0.02, aspect=28)
    cbar.set_label("mAP50–95 (%)")
    cbar.outline.set_visible(False)
    save(fig, "fault_heatmap")


def margin_matrix(table: dict, ds: str, base: str) -> np.ndarray:
    return np.array([[fault_mean(table, ds, "v2", f, s) - fault_mean(table, ds, base, f, s) for s in (1, 2, 3)]
                     for f in FAULTS])


def fig_fault_margin() -> None:
    table = load_faults()
    panels = [("m3fd", "md"), ("m3fd", "ft"), ("llvip", "md"), ("llvip", "ft")]
    mats = [margin_matrix(table, ds, base) for ds, base in panels]
    limit = 15
    norm = Normalize(vmin=-limit, vmax=limit)
    cmap = plt.get_cmap("RdBu")
    fig, axes = plt.subplots(1, 4, figsize=(WIDTH, 3.8), layout="constrained")
    for ax, (ds, base), mat, tag in zip(axes, panels, mats, "abcd"):
        im = draw_heat(ax, mat, cmap, norm, "{:+.1f}", tag == "a", cell_size=5.6)
        seeds = "three-seed mean" if ds == "m3fd" and base == "md" else "v2 mean, baseline seed 0" \
            if ds == "m3fd" else "seed 0"
        short = "concat (MD)" if base == "md" else "concat (FT)"
        ax.set_title(f"({tag}) {DS_NAME[ds]}\nv2 − {short}\n{seeds}", loc="left", fontsize=6.5)
        ax.set_xlabel("Fault severity")
    cbar = fig.colorbar(im, ax=axes, shrink=0.8, pad=0.02, aspect=28, extend="both")
    cbar.set_label("mAP margin (points), colour clipped at ±15; blue favours v2")
    cbar.outline.set_visible(False)
    save(fig, "fault_margin")


def fig_severity_curves() -> None:
    table = load_faults()
    fig, axes = plt.subplots(3, 4, figsize=(WIDTH, 5.2), sharex=True, sharey=True, layout="constrained")
    flat = axes.ravel()
    for ax, fault in zip(flat, FAULTS):
        for m in ("ft", "md", "v2"):
            sevs = range(4)
            series = [table[("m3fd", m, "clean" if s == 0 else fault, s)] for s in sevs]
            means = np.array([statistics.fmean(v) for v in series])
            stds = np.array([statistics.stdev(v) if len(v) > 1 else 0.0 for v in series])
            st = STYLE[m]
            ax.plot(list(sevs), means, color=st["color"], marker=st["marker"], markersize=3, lw=1.0, zorder=3)
            if m != "ft":
                ax.fill_between(list(sevs), means - stds, means + stds, color=st["color"], alpha=0.25, lw=0)
        ax.set_title(FAULT_LABEL[fault], fontsize=7)
        ax.set_xticks(range(4))
        ax.set_ylim(0, 60)
        ax.grid(axis="y", lw=0.3, color="#D5D9DD")
    for ax in axes[:, 0]:
        ax.set_ylabel("mAP50–95 (%)")
    for ax in axes[-1, :]:
        ax.set_xlabel("Fault severity (0 = clean)")
    legend_ax = flat[-1]
    legend_ax.axis("off")
    handles = [Line2D([], [], color=STYLE[m]["color"], marker=STYLE[m]["marker"], markersize=3.5, lw=1.0,
                      label=STYLE[m]["label"]) for m in ("v2", "md", "ft")]
    handles.append(plt.Rectangle((0, 0), 1, 1, color="#999999", alpha=0.35, lw=0, label="±1 std over seeds"))
    legend_ax.legend(handles=handles, loc="center", frameon=False, fontsize=6.5, borderaxespad=0)
    legend_ax.text(0.5, 0.12, "M3FD. v2 and modality dropout:\nthree seeds. Fault-trained: seed 0.",
                   transform=legend_ax.transAxes, ha="center", va="center", fontsize=6, color=MUTED)
    save(fig, "severity_curves")


def fig_tradeoff() -> None:
    lat = read_json(RES / "latency_local.json")
    key = {"thermal": "yolo26n_thermal", "md": "concat_md", "ft": "concat_ft", "v2": "cffm_v2", "v3": "cffm_v3"}
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.9), layout="constrained")
    for ax, ds, tag in zip(axes, ("llvip", "m3fd"), "ab"):
        for m in ("thermal", "md", "ft", "v2", "v3"):
            mean, std = mean_std(seed_values(ds, m))
            x, size = lat[key[m]]["median_ms"], lat[key[m]]["params_M"]
            st = STYLE[m]
            ax.errorbar(x, mean, yerr=std if std else None, color=st["color"], lw=0.8, capsize=2, zorder=2)
            ax.scatter(x, mean, s=size * 26, color=st["color"], marker=st["marker"], zorder=3,
                       edgecolor="white", linewidth=0.5)
        low = {"llvip": 62, "m3fd": 46}[ds]
        ax.set_ylim(low, {"llvip": 68, "m3fd": 56}[ds])
        ax.set_xlim(22, 56)
        ax.set_xlabel("Median latency, RTX 4050 (ms)")
        ax.set_ylabel(f"mAP50–95 (%), axis starts at {low}")
        ax.set_title(f"({tag}) {DS_NAME[ds]}", loc="left")
        ax.grid(lw=0.3, color="#D5D9DD")
    handles = [Line2D([], [], color=STYLE[m]["color"], marker=STYLE[m]["marker"], ls="", markersize=5,
                      label=STYLE[m]["label"]) for m in ("thermal", "md", "ft", "v2", "v3")]
    axes[0].legend(handles=handles, loc="lower right", frameon=False, fontsize=6.3, ncols=1)
    axes[1].text(0.98, 0.03, "marker area proportional to parameters (2.5 to 6.0 M)", transform=axes[1].transAxes,
                 ha="right", va="bottom", fontsize=6, color=MUTED)
    save(fig, "tradeoff")


def fig_scale() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.7), layout="constrained")
    bands = [("Small", "mAP_small(B)"), ("Medium", "mAP_medium(B)"), ("Large", "mAP_large(B)")]
    for ax, ds, tag in zip(axes, ("llvip", "m3fd"), "ab"):
        for k, (m, offset) in enumerate((("v2", -0.19), ("md", 0.19))):
            st = STYLE[m]
            for b, (_, metric) in enumerate(bands):
                vals = seed_values(ds, m, metric)
                mean, std = mean_std(vals)
                ax.bar(b + offset, mean, 0.34, color=st["color"], label=st["label"] if b == 0 else None, zorder=2)
                ax.errorbar(b + offset, mean, yerr=std, color=INK, capsize=2, lw=0.7, zorder=4)
                ax.scatter(np.full(len(vals), b + offset), vals, s=5, color="white", edgecolor=INK,
                           linewidth=0.4, zorder=5)
                ax.text(b + offset, mean + std + 1.2, f"{mean:.1f}", ha="center", va="bottom", fontsize=6)
        ax.set_xticks(range(3), [b for b, _ in bands])
        ax.set_xlabel("Object size band")
        ax.set_ylabel("AP (%)")
        ax.set_ylim(0, 100)
        ax.set_title(f"({tag}) {DS_NAME[ds]}", loc="left")
    axes[0].legend(frameon=False, loc="upper left", fontsize=6.5)
    save(fig, "scale")


def tracking_means(tracking: dict, cond: str, key: str) -> tuple[float, list[float]]:
    vals = [100 * r[key] for r in tracking[cond]]
    return statistics.fmean(vals), vals


def paired_bars(ax, groups: list[tuple[str, float, float]], width: float = 0.34, fmt: str = "{:.1f}", gap: float = 0.5) -> None:
    for i, (_, off, on) in enumerate(groups):
        ax.bar(i - width / 2 - 0.01, off, width, color=OFF_COLOR, zorder=2)
        ax.bar(i + width / 2 + 0.01, on, width, color=ON_COLOR, zorder=2)
        for x, v in ((i - width / 2, off), (i + width / 2, on)):
            ax.text(x, v + gap, fmt.format(v), ha="center", va="bottom", fontsize=5.8)
    ax.set_xticks(range(len(groups)), [g[0] for g in groups])


def fig_memory() -> None:
    mem = read_json(RES / "phase2" / "memory.json")
    trk = read_json(RES / "phase2" / "tracking.json")
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.8), layout="constrained", width_ratios=[1.55, 1])
    conds = [("Clean", "clean"), ("Thermal\nlost", "thermal lost"), ("Thermal lost,\nflagged", "thermal lost (flagged)"),
             ("Visible\ndark", "visible dark"), ("Thermal\nnoise", "thermal noise")]
    groups = [(lab, 100 * mem[f"{c} | image"]["MR_all"], 100 * mem[f"{c} | memory"]["MR_all"]) for lab, c in conds]
    ax = axes[0]
    paired_bars(ax, groups)
    ax.set_ylabel("KAIST miss rate (%), lower is better")
    ax.set_ylim(0, 58)
    ax.set_title("(a) Single-frame detection, 2262 frames", loc="left")
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=OFF_COLOR, label="Memory off (image mode)"),
                       plt.Rectangle((0, 0), 1, 1, color=ON_COLOR, label="Memory on")],
              frameon=False, loc="upper left")
    ax = axes[1]
    spec = [("MOTA\nclean", "clean", "mota"), ("MOTA\nthermal lost", "thermal lost", "mota"),
            ("IDF1\nclean", "clean", "idf1"), ("IDF1\nthermal lost", "thermal lost", "idf1")]
    width = 0.34
    for i, (lab, cond, key) in enumerate(spec):
        for off_x, mode, color in ((-width / 2 - 0.01, "image", OFF_COLOR), (width / 2 + 0.01, "memory", ON_COLOR)):
            mean, vals = tracking_means(trk, f"{cond} | {mode}", key)
            ax.bar(i + off_x, mean, width, color=color, zorder=2)
            ax.scatter(np.full(len(vals), i + off_x), vals, s=6, color="white", edgecolor=INK, linewidth=0.4, zorder=4)
            ax.text(i + off_x, 95, f"{mean:.1f}", ha="center", va="bottom", fontsize=5.8)
    ax.set_xticks(range(4), [s[0] for s in spec])
    ax.set_ylim(0, 105)
    ax.set_ylabel("Score (%), higher is better")
    ax.set_title("(b) Tracking, mean (bar) and sequences (dots)", loc="left")
    save(fig, "memory")


def fig_live() -> None:
    soak = read_json(RES / "phase3" / "soak.json")
    faults = read_json(RES / "phase3" / "faults.json")
    fig, axes = plt.subplots(1, 4, figsize=(WIDTH, 2.7), layout="constrained", width_ratios=[1.7, 0.9, 1, 1])
    rate_color = {"30 fps": "#8A929B", "20 fps": "#4A5563"}
    ax = axes[0]
    keys = [("Inference\nmedian", "infer_p50"), ("Inference\np95", "infer_p95"),
            ("End to end\nmedian", "e2e_p50"), ("End to end\np95", "e2e_p95")]
    for r, (rate, off) in enumerate((("30 fps", -0.19), ("20 fps", 0.19))):
        for i, (_, k) in enumerate(keys):
            v = soak[rate][k]
            ax.bar(i + off, v, 0.34, color=rate_color[rate], label=f"{rate} offered" if i == 0 else None, zorder=2)
            ax.text(i + off, v + 2, f"{v:.0f}", ha="center", va="bottom", fontsize=5.8)
    ax.set_xticks(range(4), [k[0] for k in keys])
    ax.set_ylabel("Delay (ms)")
    ax.set_ylim(0, 160)
    ax.legend(frameon=False, loc="upper left", fontsize=6)
    ax.set_title("(a) Soak delay", loc="left")
    ax = axes[1]
    for i, rate in enumerate(("30 fps", "20 fps")):
        fps = soak[rate]["fps"]
        ax.bar(i, fps, 0.55, color=rate_color[rate], zorder=2)
        ax.hlines(int(rate.split()[0]), i - 0.34, i + 0.34, color=INK, lw=0.9, ls=(0, (3, 1.5)), zorder=3)
        ax.text(i, fps / 2, f"{fps:.1f}", ha="center", va="center", fontsize=6, color="white")
        ax.text(i, int(rate.split()[0]) + 0.8, f"{100 * soak[rate]['capture_drop_rate']:.0f}% dropped",
                ha="center", va="bottom", fontsize=5.8)
    ax.set_xticks(range(2), [f"{r}\n({soak[r]['seconds']:.0f} s)" for r in ("30 fps", "20 fps")])
    ax.set_ylim(0, 36)
    ax.set_ylabel("Achieved frame rate (fps)")
    ax.set_title("(b) Throughput", loc="left")
    names = ["thermal unplugged", "visible frozen", "thermal stalled"]
    short = ["Thermal\nunplugged", "Visible\nfrozen", "Thermal\nstalled"]
    for ax, field, tag, head in ((axes[2], "detect_s", "c", "Detection"), (axes[3], "recover_s", "d", "Recovery")):
        groups = [(short[i], faults["monitor off"]["faults"][n][field], faults["monitor on"]["faults"][n][field])
                  for i, n in enumerate(names)]
        paired_bars(ax, groups, fmt="{:.2f}", gap=0.012)
        ax.set_ylim(0, 1.0)
        ax.set_ylabel(f"{head} time (s)")
        ax.set_title(f"({tag}) {head}", loc="left")
        ax.tick_params(axis="x", labelsize=5.8)
    axes[2].legend(handles=[plt.Rectangle((0, 0), 1, 1, color=OFF_COLOR, label="Monitor off"),
                            plt.Rectangle((0, 0), 1, 1, color=ON_COLOR, label="Monitor on")],
                   frameon=False, loc="upper left", fontsize=6)
    save(fig, "live")


def fig_multicam() -> None:
    cam = read_json(RES / "phase4" / "phase4_summary.json")
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH, 2.6), layout="constrained", width_ratios=[1.5, 1, 1])
    ax = axes[0]
    metrics = [("MODA", "MODA"), ("Precision", "precision"), ("Recall", "recall"), ("MODP", "MODP")]
    det = [("zero-shot YOLO26-m", "Zero-shot YOLO26-m", OFF_COLOR), ("fine-tuned YOLO26-m", "Fine-tuned YOLO26-m", "#3B8B66")]
    for k, (key, label, color) in enumerate(det):
        off = (-0.19, 0.19)[k]
        for i, (_, m) in enumerate(metrics):
            v = 100 * cam["detection"][key][m]
            ax.bar(i + off, v, 0.34, color=color, label=label if i == 0 else None, zorder=2)
            ax.text(i + off, v + 1.2, f"{v:.1f}", ha="center", va="bottom", fontsize=5.8)
    ax.set_xticks(range(4), [m[0] for m in metrics])
    ax.set_ylim(0, 125)
    ax.set_ylabel("Score (%)")
    ax.legend(frameon=False, loc="upper left", ncols=2, fontsize=6)
    ax.set_title("(a) Detection, ground plane", loc="left")
    ax = axes[1]
    trk = cam["tracking"]
    items = [("MOTA", 100 * trk["mota"]), ("IDF1", 100 * trk["idf1"]),
             ("Mostly\ntracked", 100 * trk["mostly_tracked"] / trk["num_unique_objects"])]
    for i, (_, v) in enumerate(items):
        ax.bar(i, v, 0.55, color="#3B8B66", zorder=2)
        ax.text(i, v + 1.2, f"{v:.1f}", ha="center", va="bottom", fontsize=5.8)
    ax.set_xticks(range(3), [n for n, _ in items])
    ax.set_ylim(0, 100)
    ax.set_ylabel("Score (%)")
    ax.set_title("(b) Tracking", loc="left")
    ax = axes[2]
    cnt = cam["counts"]
    items = [("All", cnt["count_mae"]), ("Zone 1", cnt["zone 1_mae"]), ("Zone 2", cnt["zone 2_mae"]),
             ("Zone 3", cnt["zone 3_mae"])]
    for i, (_, v) in enumerate(items):
        ax.bar(i, v, 0.55, color="#5E6C7C", zorder=2)
        ax.text(i, v + 0.03, f"{v:.2f}", ha="center", va="bottom", fontsize=5.8)
    ax.set_xticks(range(4), [n for n, _ in items])
    ax.set_ylim(0, 1.8)
    ax.set_ylabel("Count error, MAE (people)")
    ax.set_title("(c) Counting", loc="left")
    ax.text(0.99, 0.97, f"mean {cnt['mean_people']:.1f} people", transform=ax.transAxes, ha="right", va="top",
            fontsize=5.8, color=MUTED)
    save(fig, "multicam")


def read_curve(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with open(path, newline="", encoding="utf-8") as f:
        rows = [{k.strip(): v for k, v in r.items()} for r in csv.DictReader(f)]
    return (np.array([int(r["epoch"]) for r in rows]), np.array([100 * float(r["metrics/mAP50-95(B)"]) for r in rows]))


def curve_path(ds: str, model: str, seed: int) -> Path | None:
    name = "cffm_v2_e80" if model == "v2" else "concat_md_e80"
    if seed == 0:
        return RUNS / f"{ds}_{name}" / "results.csv"
    if ds == "m3fd":
        return DIST / SEED_DIR[(ds, seed)] / "runs" / f"{ds}_{name}_s{seed}" / "results.csv"
    return None


def fig_training() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.8), layout="constrained")
    for ax, ds, tag in zip(axes, ("llvip", "m3fd"), "ab"):
        inset = ax.inset_axes([0.55, 0.14, 0.41, 0.26])
        for m in ("md", "v2"):
            st = STYLE[m]
            for seed in SEEDS:
                path = curve_path(ds, m, seed)
                if path is None:
                    continue
                ep, ap = read_curve(path)
                for a, window in ((ax, slice(None)), (inset, ep >= 50)):
                    a.plot(ep[window], ap[window], color=st["color"], lw=0.8, alpha=0.9)
        ax.set_xlim(0, 80)
        ax.set_ylim(20, 70)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Validation mAP50–95 (%)")
        ax.set_title(f"({tag}) {DS_NAME[ds]}", loc="left")
        inset.set_xlim(50, 80)
        inset.tick_params(labelsize=5.5, length=2)
        inset.set_title("epochs 50 to 80", fontsize=5.8, pad=2)
        for line in ax.indicate_inset_zoom(inset, edgecolor=MUTED, lw=0.5).connectors:
            line.set_visible(False)
    handles = [Line2D([], [], color=STYLE[m]["color"], lw=1.0, label=STYLE[m]["label"]) for m in ("v2", "md")]
    axes[0].legend(handles=handles, frameon=False, loc="lower left", bbox_to_anchor=(0.02, 0.1), fontsize=6.3)
    axes[0].text(0.03, 0.03, "seed 0 only", transform=axes[0].transAxes, fontsize=6, color=MUTED)
    axes[1].text(0.03, 0.03, "seeds 0, 1, 2", transform=axes[1].transAxes, fontsize=6, color=MUTED)
    save(fig, "training")


def main() -> None:
    setup_style()
    for build in (fig_main, fig_fault_heatmap, fig_fault_margin, fig_severity_curves, fig_tradeoff, fig_scale,
                  fig_memory, fig_live, fig_multicam, fig_training):
        build()


if __name__ == "__main__":
    main()
