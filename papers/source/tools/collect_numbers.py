"""Regenerate numbers.tex and tables/*.tex from the CFFM-Net result files."""
import csv
import json
import math
import re
import statistics
import sys
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

SRC = Path(__file__).resolve().parents[1]
ROOT = SRC.parents[1]
PILOT = ROOT / "cffm-net-pilot"
RES = PILOT / "results"
RUNS = PILOT / "runs"
DIST = ROOT / "dist" / "out"
BS = "\\"

DATASETS = {"llvip": "Llvip", "m3fd": "Mfd"}
SEED = {0: "SeedZero", 1: "SeedOne", 2: "SeedTwo"}
SEV = {1: "One", 2: "Two", 3: "Three"}
SHIFT = {"4": "Four", "8": "Eight", "16": "Sixteen"}
METRICS = [
    ("Map", "mAP50-95(B)"), ("MapFifty", "mAP50(B)"),
    ("Precision", "precision(B)"), ("Recall", "recall(B)"),
    ("ApSmall", "mAP_small(B)"), ("ApMedium", "mAP_medium(B)"), ("ApLarge", "mAP_large(B)"),
    ("ApBandVeryTiny", "AP_vt(B)"), ("ApBandTiny", "AP_t(B)"), ("ApBandSmall", "AP_s(B)"),
]
SIZE = ["ApSmall", "ApMedium", "ApLarge"]
ROUND_ONE = {
    "llvip": {"yolo26n_visible": "YoloVis", "yolo26n_thermal": "YoloThm", "concat": "ConcatPlain",
              "cffm": "CffmVone", "cffm_nogate": "CffmVoneNogate", "cffm_gconv": "CffmVoneGconv",
              "cffm_nop2b": "CffmVoneNoptwo", "concat_p2": "ConcatPtwo"},
    "m3fd": {"concat": "ConcatPlain", "cffm": "CffmVone", "yolo26n_visible_b32": "YoloVis",
             "yolo26n_thermal_b32": "YoloThm"},
}
V3 = {"cffm_v3_e80": "CffmVthree", "cffm_v3_norel_e80": "CffmVthreeNorel", "concat_ft_e80": "ConcatFt"}
V3_DIR = {"llvip": "08k-v3-llvip", "m3fd": "08l-v3-m3fd"}
SEED_DIR = {("llvip", 1): "08g", ("llvip", 2): "08h", ("m3fd", 1): "08i", ("m3fd", 2): "08j"}
E80 = {"cffm_v2_e80": "CffmVtwo", "concat_md_e80": "ConcatMd"}
LABEL = {
    "YoloVis": "YOLO26-n visible", "YoloThm": "YOLO26-n thermal", "ConcatPlain": "Two-stream concat",
    "CffmVone": "CFFM-Net v1", "CffmVoneNogate": "v1, no gate", "CffmVoneGconv": "v1, gated conv",
    "CffmVoneNoptwo": "v1, no P2 head", "ConcatPtwo": "Concat + P2 head",
    "ConcatMd": "Concat, modality dropout", "CffmVtwo": "CFFM-Net v2", "ConcatFt": "Concat, fault training",
    "CffmVthree": "CFFM-Net v3", "CffmVthreeNorel": "v3, no reliability branch",
}
DS_NAME = {"llvip": "LLVIP", "m3fd": "M3FD"}
FAULTS = ["visible_dark", "visible_glare", "visible_fog", "visible_noise", "visible_blur", "visible_drop",
          "thermal_drop", "thermal_shift", "thermal_noise", "thermal_crossover", "thermal_dead"]
FAULT_LABEL = {f: f.replace("_", " ").capitalize() for f in FAULTS}
MARGINS = [
    ("VtwoVsConcatMd", "CffmVtwo", "ConcatMd", ["llvip", "m3fd"], [0, 1, 2]),
    ("VthreeVsConcatFt", "CffmVthree", "ConcatFt", ["llvip", "m3fd"], [0]),
    ("VthreeVsVthreeNorel", "CffmVthree", "CffmVthreeNorel", ["llvip", "m3fd"], [0]),
    ("VthreeVsVtwo", "CffmVthree", "CffmVtwo", ["llvip", "m3fd"], [0]),
    ("VoneVsConcatPlain", "CffmVone", "ConcatPlain", ["llvip", "m3fd"], [0]),
    ("VoneVsVoneNogate", "CffmVone", "CffmVoneNogate", ["llvip"], [0]),
    ("VoneVsVoneGconv", "CffmVone", "CffmVoneGconv", ["llvip"], [0]),
    ("VoneNoptwoVsConcatPtwo", "CffmVoneNoptwo", "ConcatPtwo", ["llvip"], [0]),
    ("VoneVsConcatPtwo", "CffmVone", "ConcatPtwo", ["llvip"], [0]),
]

M: dict[str, str] = {}
FLT: dict = {}
GROUPS: list[tuple[str, list[str]]] = []
CHECKS: list[str] = []


def q(x: float, nd: int, scale: int = 1) -> str:
    """Format x*scale to nd decimals, round half up."""
    d = Decimal(repr(float(x))) * scale
    return str(d.quantize(Decimal(1).scaleb(-nd), rounding=ROUND_HALF_UP))


def signed(x: float, nd: int = 1, scale: int = 100) -> str:
    """Format a signed difference with a math sign."""
    s = q(abs(x), nd, scale)
    if float(s) == 0:
        return s
    return (BS + "ensuremath{+}" if x > 0 else BS + "ensuremath{-}") + s


def group(comment: str) -> list[str]:
    """Open a macro group with a one-line comment."""
    names: list[str] = []
    GROUPS.append((comment, names))
    return names


def put(names: list[str], key: str, value: str) -> None:
    """Register one macro."""
    assert key.isalpha(), key
    assert key not in M, key
    M[key] = value
    names.append(key)


def load(path: Path) -> dict:
    """Read a JSON file."""
    return json.loads(path.read_text(encoding="utf-8"))


def finite(v) -> bool:
    """True for a real number."""
    return isinstance(v, (int, float)) and math.isfinite(v)


def read_csv(p: Path) -> list[dict]:
    """Read a CSV file into dict rows."""
    with p.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def camel(s: str) -> str:
    """Convert snake_case to CamelCase."""
    return "".join(w.capitalize() for w in s.split("_"))


def run_sources() -> dict[tuple[str, str, int], Path]:
    """Map (dataset, model macro, seed) to its metrics.json."""
    s: dict[tuple[str, str, int], Path] = {}
    for ds, models in ROUND_ONE.items():
        for r, name in models.items():
            s[(ds, name, 0)] = RUNS / "eval" / f"{ds}_{r}" / "metrics.json"
    for ds in DATASETS:
        for r, name in E80.items():
            s[(ds, name, 0)] = RUNS / "faults" / f"{ds}_{r}" / "clean" / "metrics.json"
            for seed in (1, 2):
                s[(ds, name, seed)] = (DIST / SEED_DIR[(ds, seed)] / "runs" / "eval"
                                       / f"{ds}_{r}_s{seed}" / "metrics.json")
        for r, name in V3.items():
            s[(ds, name, 0)] = DIST / V3_DIR[ds] / "runs" / "eval" / f"{ds}_{r}" / "metrics.json"
    return s


def results_block() -> dict[tuple[str, str, int], dict]:
    """Emit per-run metric macros, seed means and margins."""
    data = {}
    for (ds, model, seed), p in sorted(run_sources().items()):
        if not p.exists():
            CHECKS.append(f"MISSING {p}")
            continue
        data[(ds, model, seed)] = load(p)
    for (ds, model, seed), m in data.items():
        names = group(f"results: {DS_NAME[ds]} {LABEL[model]} seed {seed}, percent")
        for tok, key in METRICS:
            if finite(m.get(key)):
                put(names, f"res{DATASETS[ds]}{model}{SEED[seed]}{tok}", q(m[key], 1, 100))
    for ds in DATASETS:
        for model in E80.values():
            have = [s for s in (0, 1, 2) if (ds, model, s) in data]
            names = group(f"seed statistics: {DS_NAME[ds]} {LABEL[model]}, mean and sample std in percent")
            for tok, key in METRICS:
                vals = [data[(ds, model, s)][key] for s in have if finite(data[(ds, model, s)].get(key))]
                if len(vals) == len(have) and len(vals) > 1:
                    put(names, f"res{DATASETS[ds]}{model}Mean{tok}", q(statistics.mean(vals), 1, 100))
                    put(names, f"res{DATASETS[ds]}{model}Std{tok}", q(statistics.stdev(vals), 1, 100))
            put(names, f"res{DATASETS[ds]}{model}SeedCount", str(len(have)))
    for tag, a, b, dss, seeds in MARGINS:
        for ds in dss:
            names = group(f"margins: {DS_NAME[ds]} {tag}, percentage points")
            have = [s for s in seeds if (ds, a, s) in data and (ds, b, s) in data]
            for tok, key in METRICS[:2] + METRICS[4:7]:
                diffs = [data[(ds, a, s)][key] - data[(ds, b, s)][key] for s in have]
                for s, d in zip(have, diffs):
                    put(names, f"mar{DATASETS[ds]}{tag}{SEED[s]}{tok}", signed(d))
                if len(diffs) > 1:
                    put(names, f"mar{DATASETS[ds]}{tag}Mean{tok}", signed(statistics.mean(diffs)))
                    put(names, f"mar{DATASETS[ds]}{tag}Std{tok}", q(statistics.stdev(diffs), 1, 100))
    return data


FAULT_MODELS = {"cffm_v2": "CffmVtwo", "concat_md": "ConcatMd", "concat_ft": "ConcatFt"}
FAULT_RUN = re.compile(r"^(llvip|m3fd)_(cffm_v2|concat_md|concat_ft)_e80(?:_s(\d))?$")
T975_DF2 = 4.303


def spread(vals: list[float]) -> tuple[float, float]:
    """Mean and sample std; std is zero for one value."""
    return statistics.mean(vals), (statistics.stdev(vals) if len(vals) > 1 else 0.0)


def put_seeds(names: list[str], stem: str, tok: str, vals: list[float], nd: int = 1, scale: int = 100,
              sign: bool = False) -> None:
    """One value for a single seed, mean and std (tokens Mean and Std) for several."""
    f = (lambda x: signed(x, nd, scale)) if sign else (lambda x: q(x, nd, scale))
    if len(vals) == 1:
        put(names, stem + tok, f(vals[0]))
    else:
        mean, sd = spread(vals)
        put(names, f"{stem}Mean{tok}", f(mean))
        put(names, f"{stem}Std{tok}", q(sd, nd, scale))


def ttest_df2(diffs: list[float]) -> tuple[float, float, float]:
    """Paired t-test on three differences: two-sided p, 95 percent half-width and mean."""
    assert len(diffs) == 3
    mean, sd = spread(diffs)
    if sd == 0:
        return (1.0 if mean == 0 else 0.0), 0.0, mean
    t = abs(mean) / (sd / math.sqrt(3))
    p = 1 - t / math.sqrt(t * t + 2)
    return p, T975_DF2 * sd / math.sqrt(3), mean


def faults_block() -> None:
    """Emit fault benchmark macros: one seed gives values, several give mean and std over seeds."""
    tab: dict[tuple, dict] = {}
    for r in read_csv(RES / "faults.csv"):
        m = FAULT_RUN.match(r["run"])
        assert m, r["run"]
        ds, model, seed = m.group(1), FAULT_MODELS[m.group(2)], int(m.group(3) or 0)
        tab[(ds, model, seed, r["fault"], int(r["severity"]), r["flagged"] == "True")] = r
    seeds = {(d, m): sorted({k[2] for k in tab if k[:2] == (d, m)}) for d in DATASETS for m in FAULT_MODELS.values()}
    FLT.update({"tab": tab, "seeds": seeds})

    def vals(ds, model, fault, sev, flag, col="AP"):
        return [float(tab[(ds, model, s, fault, sev, flag)][col]) for s in seeds[(ds, model)]]

    for ds in DATASETS:
        for model in FAULT_MODELS.values():
            n = len(seeds[(ds, model)])
            names = group(f"fault benchmark: {DS_NAME[ds]} {LABEL[model]}, {n} seed(s), percent")
            first = seeds[(ds, model)][0]
            for (d, m, s, fault, sev, flag) in sorted(k for k in tab if k[:3] == (ds, model, first)):
                stem = (f"flt{DATASETS[ds]}{model}Clean" if fault == "clean"
                        else f"flt{DATASETS[ds]}{model}{camel(fault)}{SEV[sev]}{'Flagged' if flag else ''}")
                put_seeds(names, stem, "Map", vals(ds, model, fault, sev, flag))
                put_seeds(names, stem, "MapFifty", vals(ds, model, fault, sev, flag, "AP50"))
            put(names, f"flt{DATASETS[ds]}{model}SeedCount", str(n))
    summary = {}
    for ds in DATASETS:
        for model in FAULT_MODELS.values():
            names = group(f"fault summary at severity three: {DS_NAME[ds]} {LABEL[model]}, percent")
            per = []
            for s in seeds[(ds, model)]:
                sev3 = {f: float(tab[(ds, model, s, f, 3, False)]["AP"]) for f in FAULTS}
                clean = float(tab[(ds, model, s, "clean", 0, False)]["AP"])
                mean = statistics.mean(sev3.values())
                per.append({"sev3": sev3, "clean": clean, "mean": mean,
                            "vis": statistics.mean(v for f, v in sev3.items() if f.startswith("visible")),
                            "thm": statistics.mean(v for f, v in sev3.items() if f.startswith("thermal"))})
            stem = f"flt{DATASETS[ds]}{model}"
            put_seeds(names, stem, "AvgSevThreeMap", [p["mean"] for p in per])
            put_seeds(names, stem, "VisibleAvgSevThreeMap", [p["vis"] for p in per])
            put_seeds(names, stem, "ThermalAvgSevThreeMap", [p["thm"] for p in per])
            put_seeds(names, stem, "DropAvgSevThreeMap", [p["clean"] - p["mean"] for p in per])
            put_seeds(names, stem, "RetainedSevThreePercent", [p["mean"] / p["clean"] for p in per])
            avg = {f: statistics.mean(p["sev3"][f] for p in per) for f in FAULTS}
            worst = min(avg, key=avg.get)
            put(names, stem + "WorstSevThreeMap", q(avg[worst], 1, 100))
            put(names, stem + "WorstSevThreeFault", FAULT_LABEL[worst])
            summary[(ds, model)] = per
    for ds in DATASETS:
        names = group(f"fault comparison: {DS_NAME[ds]} CFFM-Net v2 against concat at severity three, percentage points")
        a, b = summary[(ds, "CffmVtwo")], summary[(ds, "ConcatMd")]
        assert len(a) == len(b)
        n = len(a)
        stem = f"flt{DATASETS[ds]}"
        pairs = list(zip(a, b))
        put_seeds(names, stem + "Margin", "AvgSevThreeMap", [x["mean"] - y["mean"] for x, y in pairs], sign=True)
        put_seeds(names, stem + "Margin", "VisibleAvgSevThreeMap", [x["vis"] - y["vis"] for x, y in pairs], sign=True)
        put_seeds(names, stem + "Margin", "ThermalAvgSevThreeMap", [x["thm"] - y["thm"] for x, y in pairs], sign=True)
        put_seeds(names, stem + "Margin", "RetainedSevThreePercent",
                  [x["mean"] / x["clean"] - y["mean"] / y["clean"] for x, y in pairs], sign=True)
        if n == 3:
            p, half, mean = ttest_df2([x["mean"] - y["mean"] for x, y in pairs])
            put(names, stem + "MarginAvgP", q(p, 2))
            put(names, stem + "MarginAvgCiLow", signed(mean - half))
            put(names, stem + "MarginAvgCiHigh", signed(mean + half))
        avg_m = {f: statistics.mean(x["sev3"][f] - y["sev3"][f] for x, y in pairs) for f in FAULTS}
        put(names, stem + "WinsSevThree", str(sum(v > 0 for v in avg_m.values())))
        put(names, stem + "FaultCount", str(len(FAULTS)))
        if n > 1:
            agree = sum(len({(x["sev3"][f] - y["sev3"][f]) > 0 for x, y in pairs}) == 1 for f in FAULTS)
            put(names, stem + "SignAgreeSevThree", str(agree))
        for f in FAULTS:
            put_seeds(names, f"{stem}Margin{camel(f)}SevThree", "Map", [x["sev3"][f] - y["sev3"][f] for x, y in pairs],
                      sign=True)
        flag = {(f, m): statistics.mean(vals(ds, m, f, 3, True)) for f in ("visible_drop", "thermal_drop")
                for m in ("CffmVtwo", "ConcatMd")}
        for f in ("visible_drop", "thermal_drop"):
            put(names, f"{stem}Margin{camel(f)}FlaggedSevThreeMap", signed(flag[(f, "CffmVtwo")] - flag[(f, "ConcatMd")]))
    for ds in DATASETS:
        names = group(f"fault comparison: {DS_NAME[ds]} CFFM-Net v2 against fault-trained concat, seed zero, severity three, pp")
        a, b = summary[(ds, "CffmVtwo")][0], summary[(ds, "ConcatFt")][0]
        stem = f"flt{DATASETS[ds]}VtwoVsConcatFt"
        put(names, stem + "AvgSevThreeMap", signed(a["mean"] - b["mean"]))
        put(names, stem + "GapSevThreeMap", q(abs(a["mean"] - b["mean"]), 1, 100))
        put(names, stem + "CleanMap", signed(a["clean"] - b["clean"]))
        put(names, stem + "Wins", str(sum(a["sev3"][f] > b["sev3"][f] for f in FAULTS)))
        put(names, f"flt{DATASETS[ds]}ConcatMdVsConcatFtAvgSevThreeMap",
            signed(summary[(ds, "ConcatMd")][0]["mean"] - b["mean"]))


def latency_block() -> None:
    """Emit parameters, FLOPs and latency macros."""
    lat = load(RUNS / "latency" / "latency.json")
    if lat != load(RES / "pilot_v1" / "latency.json"):
        CHECKS.append("DISAGREE runs/latency/latency.json vs results/pilot_v1/latency.json")
    inv = {f"{ds}_{r}": (ds, n) for ds, ms in ROUND_ONE.items() for r, n in ms.items()}
    for run, v in lat.items():
        ds, model = inv[run]
        names = group(f"latency: {DS_NAME[ds]} {LABEL[model]}, parameters in millions, GFLOPs, milliseconds")
        stem = f"lat{DATASETS[ds]}{model}"
        put(names, stem + "ParamsM", q(v["params_M"], 2))
        put(names, stem + "Gflops", q(v["GFLOPs"], 1))
        put(names, stem + "MedianMs", q(v["median_ms"], 1))
        put(names, stem + "PNinetyFiveMs", q(v["p95_ms"], 1))
    for ds in DATASETS:
        names = group(f"latency ratios: {DS_NAME[ds]} CFFM-Net v1 over two-stream concat")
        a, b = lat[f"{ds}_cffm"], lat[f"{ds}_concat"]
        put(names, f"lat{DATASETS[ds]}RatioMedian", q(a["median_ms"] / b["median_ms"], 2))
        put(names, f"lat{DATASETS[ds]}RatioParams", q(a["params_M"] / b["params_M"], 2))
        put(names, f"lat{DATASETS[ds]}RatioGflops", q(a["GFLOPs"] / b["GFLOPs"], 2))
    for r in read_csv(RES / "main_results.csv"):
        if r["params_M"] and r["run"] in lat:
            for col in ("params_M", "GFLOPs", "median_ms"):
                if abs(float(r[col]) - lat[r["run"]][col]) > 5e-4:
                    CHECKS.append(f"DISAGREE main_results.csv {r['run']} {col}")


def probe_token(label: str) -> str:
    """Turn a probe row label into a macro token."""
    flagged = label.endswith("(flagged)")
    base = label.replace(" (flagged)", "")
    if base.startswith("thermal_shift_"):
        base = "thermal_shift_" + SHIFT[base.rsplit("_", 1)[1]]
    return camel(base) + ("Flagged" if flagged else "")


def probe_block() -> None:
    """Emit degradation-probe macros."""
    rows = read_csv(RES / "probe.csv")
    if rows != read_csv(RES / "pilot_v1" / "probe.csv"):
        CHECKS.append("DISAGREE results/probe.csv vs results/pilot_v1/probe.csv")
    cols = {"llvip_cffm": "CffmVone", "llvip_cffm_nogate": "CffmVoneNogate", "llvip_concat": "ConcatPlain"}
    names = group("degradation probe on LLVIP, round-one models, mAP50-95 in percent")
    for r in rows:
        for col, model in cols.items():
            put(names, f"prbLlvip{model}{probe_token(r['probe'])}Map", q(float(r[col]), 1, 100))


def hyp_block() -> None:
    """Emit the pilot hypothesis verdicts."""
    words = {"1": "One", "2": "Two", "3": "Three", "4": "Four", "5": "Five"}
    if read_csv(RES / "hypotheses.csv") != read_csv(RES / "pilot_v1" / "hypotheses.csv"):
        CHECKS.append("DISAGREE results/hypotheses.csv vs results/pilot_v1/hypotheses.csv")
    names = group("pilot hypotheses from round one: statement, evidence, verdict")
    for r in read_csv(RES / "hypotheses.csv"):
        stem = "hypPh" + words[r["id"][2:]]
        for k, tok in (("hypothesis", "Statement"), ("evidence", "Evidence"), ("verdict", "Verdict")):
            put(names, stem + tok, r[k].replace("_", BS + "_"))


def phase_block() -> None:
    """Emit phase four detection, tracking and counting macros."""
    j = load(RES / "phase4" / "phase4_summary.json")
    zone = {"zone 1": "ZoneOne", "zone 2": "ZoneTwo", "zone 3": "ZoneThree"}
    det = {"zero-shot YOLO26-m": "ZeroShot", "fine-tuned YOLO26-m": "FineTuned"}
    names = group("multi-camera detection: MODA, MODP, precision, recall in percent, counts as integers")
    for k, tok in det.items():
        d = j["detection"][k]
        for f, t in (("MODA", "Moda"), ("MODP", "Modp"), ("precision", "Precision"), ("recall", "Recall")):
            put(names, f"camDet{tok}{t}", q(d[f], 1, 100))
        for f in ("TP", "FP", "FN"):
            put(names, f"camDet{tok}{f.capitalize()}", str(d[f]))
        p = j["fusion_params"][k]
        put(names, f"camFusion{tok}Radius", str(p["radius"]))
        put(names, f"camFusion{tok}MinRatio", q(p["min_ratio"], 2))
        put(names, f"camFusion{tok}MinScore", q(p["min_score"], 2))
    t = j["tracking"]
    names = group("multi-camera tracking: MOTA and IDF1 in percent, switches and tracks as integers")
    put(names, "camTrackMota", q(t["mota"], 1, 100))
    put(names, "camTrackIdfOne", q(t["idf1"], 1, 100))
    put(names, "camTrackSwitches", str(int(t["num_switches"])))
    put(names, "camTrackMostlyTracked", str(int(t["mostly_tracked"])))
    put(names, "camTrackUniqueObjects", str(int(t["num_unique_objects"])))
    c = j["counts"]
    names = group("multi-camera counting: mean absolute error in people, relative error in percent")
    put(names, "camCountMae", q(c["count_mae"], 2))
    put(names, "camCountMeanPeople", q(c["mean_people"], 1))
    put(names, "camCountRelErr", q(c["count_rel_err"], 1, 100))
    for z, tok in zone.items():
        put(names, f"camCount{tok}Mae", q(c[f"{z}_mae"], 2))
    names = group("multi-camera line crossings and zone dwell times in seconds")
    put(names, "camCrossForward", str(j["crossings of y = 0"]["forward"]))
    put(names, "camCrossBackward", str(j["crossings of y = 0"]["backward"]))
    for z, tok in zone.items():
        d = j["dwell"][z]
        put(names, f"camDwell{tok}Tracks", str(d["tracks"]))
        put(names, f"camDwell{tok}TotalS", q(d["total_s"], 1))
        put(names, f"camDwell{tok}MeanS", q(d["mean_s"], 1))


def cross_checks(data) -> None:
    """Compare the summary CSVs against the metrics files."""
    if read_csv(RES / "main_results.csv") != read_csv(RES / "pilot_v1" / "main_results.csv"):
        CHECKS.append("DISAGREE results/main_results.csv vs results/pilot_v1/main_results.csv")
    for r in read_csv(RES / "main_results.csv"):
        ds, rest = r["run"].split("_", 1)
        name = ROUND_ONE[ds].get(rest)
        if name is None or (ds, name, 0) not in data:
            continue
        m = data[(ds, name, 0)]
        for col, key in (("AP", "mAP50-95(B)"), ("AP50", "mAP50(B)"), ("AP_small", "mAP_small(B)")):
            if abs(float(r[col]) - m[key]) > 5.1e-5:
                CHECKS.append(f"DISAGREE main_results.csv {r['run']} {col} {r[col]} vs metrics {m[key]:.4f}")
    seen: dict[tuple, set] = {}
    for r in read_csv(RES / "faults.csv"):
        ds = r["dataset"]
        if r["fault"] == "clean":
            mt = FAULT_RUN.match(r["run"])
            m = data[(ds, FAULT_MODELS[mt.group(2)], int(mt.group(3) or 0))]["mAP50-95(B)"]
            if abs(float(r["AP"]) - m) > 3e-4:
                CHECKS.append(f"DISAGREE faults.csv clean {r['run']} {r['AP']} vs metrics {m:.4f}")
        seen.setdefault((r["run"], r["fault"], r["flagged"]), set()).add(r["AP"])
    for (run, fault, fl), aps in sorted(seen.items()):
        if fault.endswith("_drop") and fl == "False" and len(aps) == 1:
            CHECKS.append(f"NOTE {run} {fault}: severities one to three give identical AP")


def write_numbers() -> None:
    """Write numbers.tex."""
    lines = ["% generated by tools/collect_numbers.py; do not edit by hand"]
    for comment, names in GROUPS:
        if names:
            lines.append(f"% {comment}")
            lines += [f"{BS}newcommand{{{BS}{n}}}{{{M[n]}}}" for n in names]
    (SRC / "numbers.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def ms(stem: str, tok: str) -> str:
    """Return a mean plus or minus std cell when the stem has seeds, else the single value."""
    if f"{stem}Mean{tok}" in M:
        return pm(f"{stem}Mean{tok}", f"{stem}Std{tok}")
    return cell(stem + tok)


def cell(name: str) -> str:
    """Return a macro call, or a dash when it does not exist."""
    return BS + name if name in M else "--"


def pm(a: str, b: str) -> str:
    """Return mean plus or minus std from two macros."""
    if a in M and b in M:
        return f"{cell(a)}{BS},{BS}ensuremath{{{BS}pm}}{BS},{cell(b)}"
    return cell(a)


def table(name: str, caption: str, label: str, spec: str, header: str, body: list[str], star: bool = False,
          append: bool = False) -> None:
    """Write one booktabs table file; star makes it span both columns, append adds it to an existing file."""
    env = "table*" if star else "table"
    out = [f"{BS}begin{{{env}}}[t]", f"{BS}centering", f"{BS}caption{{{caption}}}", f"{BS}label{{{label}}}",
           f"{{{BS}footnotesize{BS}setlength{{{BS}tabcolsep}}{{4pt}}", f"{BS}begin{{tabular}}{{{spec}}}",
           f"{BS}toprule", header + f" {BS}{BS}", f"{BS}midrule"]
    for row in body:
        out.append(f"{BS}midrule" if row == "-" else row + f" {BS}{BS}")
    out += [f"{BS}bottomrule", f"{BS}end{{tabular}}}}", f"{BS}end{{{env}}}"]
    path = SRC / "tables" / f"{name}.tex"
    text = "\n".join(out) + "\n"
    path.write_text((path.read_text(encoding="utf-8") if append else "") + text, encoding="utf-8")


def sub(title: str, cols: int) -> str:
    """Return a spanning sub-header row."""
    return f"{BS}multicolumn{{{cols}}}{{l}}{{{BS}emph{{{title}}}}}"


def span(macro: str) -> str:
    """Return a two-column cell holding one macro."""
    return f"{BS}multicolumn{{2}}{{c}}{{{cell(macro)}}}"


def stat(ds: str, model: str, tok: str) -> str:
    """Return a mean plus or minus std cell."""
    return pm(f"res{DATASETS[ds]}{model}Mean{tok}", f"res{DATASETS[ds]}{model}Std{tok}")


def main_tables() -> None:
    """Write the main, scale, ablation and round tables."""
    body = []
    for i, ds in enumerate(DATASETS):
        if i:
            body.append("-")
        body.append(sub(DS_NAME[ds], 3))
        for model in ("YoloVis", "YoloThm"):
            s = f"res{DATASETS[ds]}{model}SeedZero"
            if s + "Map" in M:
                body.append(f"{LABEL[model]} & {cell(s + 'Map')} & {cell(s + 'MapFifty')}")
        for model in ("ConcatMd", "CffmVtwo"):
            body.append(f"{LABEL[model]} & {stat(ds, model, 'Map')} & {stat(ds, model, 'MapFifty')}")
        mg = f"mar{DATASETS[ds]}VtwoVsConcatMd"
        body.append(f"Margin of v2 over concat (pp) & {pm(mg + 'MeanMap', mg + 'StdMap')} & "
                    f"{pm(mg + 'MeanMapFifty', mg + 'StdMapFifty')}")
        if ds == "m3fd":
            g = "gainMfdCffmVtwoOverBest"
            body.append(f"Margin of v2 over best single camera (pp) & {pm(g + 'MeanMap', g + 'StdMap')} & "
                        f"{pm(g + 'MapFiftyMean', g + 'MapFiftyStd')}")
    table("tab_main", "Detection accuracy in percent. Fusion models: mean and sample std over three seeds, 80 epochs. "
          "Single-camera rows: one seed; 30 epochs on LLVIP, which is not matched to the fusion rows, and 80 epochs on M3FD, "
          "which is. The margin over the best single camera subtracts the one-seed single-camera value from each fusion seed.",
          "tab:main", "lcc", "Model & mAP & mAP$_{50}$", body)
    body = []
    for i, ds in enumerate(DATASETS):
        if i:
            body.append("-")
        body.append(sub(DS_NAME[ds], 4))
        for model in ("ConcatMd", "CffmVtwo"):
            body.append(f"{LABEL[model]} & " + " & ".join(stat(ds, model, t) for t in SIZE))
        mg = f"mar{DATASETS[ds]}VtwoVsConcatMd"
        body.append("Margin of v2 (pp) & " + " & ".join(pm(f"{mg}Mean{t}", f"{mg}Std{t}") for t in SIZE))
    table("tab_scale", "Average precision by object size in percent, mean and sample std over three seeds, 80 epochs.",
          "tab:scale", "lccc", "Model & Small & Medium & Large", body)
    body = [sub("Round one, 30 epochs, seed 0", 3)]
    for model in ("ConcatPlain", "CffmVone", "CffmVoneNogate", "CffmVoneGconv", "ConcatPtwo", "CffmVoneNoptwo"):
        body.append(f"{LABEL[model]} & " + " & ".join(cell(f"res{DATASETS[d]}{model}SeedZeroMap") for d in DATASETS))
    body += ["-", sub("Rounds three and four, 80 epochs, seed 0", 3)]
    for model in ("ConcatMd", "CffmVtwo", "ConcatFt", "CffmVthree", "CffmVthreeNorel"):
        body.append(f"{LABEL[model]} & " + " & ".join(cell(f"res{DATASETS[d]}{model}SeedZeroMap") for d in DATASETS))
    table("tab_ablation", "Ablation of mAP in percent. A dash marks a run that was not made.", "tab:ablation",
          "lcc", "Variant & LLVIP & M3FD", body)
    rounds = [
        ("Round one, v1 against concat (30 epochs)", "VoneVsConcatPlain", "SeedZero"),
        ("Head control, v1 without P2 against concat with P2", "VoneNoptwoVsConcatPtwo", "SeedZero"),
        ("Round three, v2 against concat (seed 0)", "VtwoVsConcatMd", "SeedZero"),
        ("Round three, v2 against concat (three-seed mean)", "VtwoVsConcatMd", "Mean"),
        ("Round four, v3 against fault-trained concat", "VthreeVsConcatFt", "SeedZero"),
        ("Round four, v3 against v3 without reliability branch", "VthreeVsVthreeNorel", "SeedZero"),
        ("Round four, v3 against v2", "VthreeVsVtwo", "SeedZero"),
    ]
    body = [f"{t} & " + " & ".join(cell(f"mar{DATASETS[d]}{tag}{s}Map") for d in DATASETS) for t, tag, s in rounds]
    table("tab_rounds", "Round history: mAP margin of CFFM-Net over its control in percentage points.", "tab:rounds",
          f"p{{0.6{BS}columnwidth}}cc", "Comparison & LLVIP & M3FD", body)


def fault_table() -> None:
    """Write the fault robustness table."""
    models = ("CffmVtwo", "ConcatMd", "ConcatFt")
    head = (f"& {BS}multicolumn{{3}}{{c}}{{LLVIP, seed 0}} & {BS}multicolumn{{3}}{{c}}{{M3FD, mean {BS}ensuremath{{{BS}pm}} std, "
            f"three seeds}} {BS}{BS} {BS}cmidrule(lr){{2-4}}{BS}cmidrule(lr){{5-7}} Fault & v2 & Concat & Concat FT & "
            f"v2 & Concat & Concat FT")

    def row(label: str, tail: str, tok: str) -> str:
        return f"{label} & " + " & ".join(ms(f"flt{DATASETS[d]}{m}{tail}", tok) for d in DATASETS for m in models)

    body = [row("Clean", "Clean", "Map"), "-"]
    for f in FAULTS:
        body.append(row(FAULT_LABEL[f], f"{camel(f)}Three", "Map"))
    for f in ("visible_drop", "thermal_drop"):
        body.append(row(f"{FAULT_LABEL[f]}, flagged", f"{camel(f)}ThreeFlagged", "Map"))
    body += ["-", row("Mean over faults", "", "AvgSevThreeMap"), row(f"Retained ({BS}%)", "", "RetainedSevThreePercent")]
    table("tab_fault", "Fault benchmark at severity three, mAP in percent, 80 epochs. LLVIP is seed 0 only. On M3FD the v2 and "
          "concat columns (modality dropout) are mean and sample std over three seeds, and the fault-trained concat column "
          "(Concat FT) is seed 0. Flagged rows tell the model which camera dropped.", "tab:fault", "lcccccc", head, body,
          star=True)


def probe_table() -> None:
    """Write the degradation probe table."""
    conds = ["Clean", "VisibleDark", "VisibleDrop", "VisibleDropFlagged", "ThermalDrop", "ThermalDropFlagged",
             "ThermalShiftFour", "ThermalShiftEight", "ThermalShiftSixteen"]
    shown = ["Clean", "Visible dark", "Visible drop", "Visible drop, flagged", "Thermal drop", "Thermal drop, flagged",
             "Thermal shift 4 px", "Thermal shift 8 px", "Thermal shift 16 px"]
    models = ["CffmVone", "CffmVoneNogate", "ConcatPlain"]
    body = [f"{s} & " + " & ".join(cell(f"prbLlvip{m}{c}Map") for m in models) for c, s in zip(conds, shown)]
    table("tab_probe", "Degradation probe on LLVIP, mAP in percent, round-one models (30 epochs, seed 0).",
          "tab:probe", "lccc", "Condition & v1 & v1, no gate & Concat", body)


def latency_table() -> None:
    """Write the latency table from the local re-timing."""
    models = ["YoloThm", "ConcatMd", "ConcatFt", "Vtwo", "VthreeNorel", "Vthree"]
    names = {"YoloThm": "YOLO26-n thermal only", "ConcatMd": LABEL["ConcatMd"], "ConcatFt": LABEL["ConcatFt"],
             "Vtwo": "CFFM-Net v2", "VthreeNorel": LABEL["CffmVthreeNorel"], "Vthree": "CFFM-Net v3"}
    body = [f"{names[m]} & " + " & ".join(cell(f"latLocal{m}{t}") for t in ("ParamsM", "Gflops", "MedianMs", "PNinetyFiveMs"))
            for m in models]
    body += ["-", "v2 over concat, ratio & " + " & ".join(cell(f"latLocalRatio{t}") for t in ("Params", "Gflops", "Median",
                                                                                              "PNinetyFive"))]
    table("tab_latency", "Parameters (millions), GFLOPs and latency (ms) on a \\latLocalDevice{}, half precision, batch one, "
          "\\cfgLatW{}$\\times$\\cfgLatH{} input, median and 95th percentile per image over \\cfgLatIters{} timed iterations "
          "after \\cfgLatWarmup{} warm-up iterations, on an otherwise idle GPU.", "tab:latency", "lrrrr",
          "Model & Params & GFLOPs & Median & p95", body)


def camera_table() -> None:
    """Write the multi-camera table."""
    body = [sub("Detection", 3)]
    for t, lab in (("Moda", f"MODA ({BS}%)"), ("Modp", f"MODP ({BS}%)"), ("Precision", f"Precision ({BS}%)"),
                   ("Recall", f"Recall ({BS}%)"), ("Tp", "True positives"), ("Fp", "False positives"),
                   ("Fn", "False negatives")):
        body.append(f"{lab} & {cell('camDetZeroShot' + t)} & {cell('camDetFineTuned' + t)}")
    body += ["-", sub("Tracking", 3),
             f"MOTA ({BS}%) & {span('camTrackMota')}", f"IDF1 ({BS}%) & {span('camTrackIdfOne')}",
             f"ID switches & {span('camTrackSwitches')}",
             f"Mostly tracked of {cell('camTrackUniqueObjects')} & {span('camTrackMostlyTracked')}",
             "-", sub("Counting", 3), f"Count MAE (people) & {span('camCountMae')}",
             f"Relative error ({BS}%) & {span('camCountRelErr')}",
             f"Zone one MAE & {span('camCountZoneOneMae')}", f"Zone two MAE & {span('camCountZoneTwoMae')}",
             f"Zone three MAE & {span('camCountZoneThreeMae')}"]
    table("tab_multicam", "Multi-camera analytics. Detection uses zero-shot and fine-tuned YOLO26-m with ground-plane "
          "fusion; tracking and counting are reported once.", "tab:multicam", "lcc",
          "Measure & Zero-shot & Fine-tuned", body)


LAT_LOCAL = {"yolo26n_thermal": "YoloThm", "concat_md": "ConcatMd", "concat_ft": "ConcatFt", "cffm_v2": "Vtwo",
             "cffm_v3_norel": "VthreeNorel", "cffm_v3": "Vthree"}
COND = {"clean": "Clean", "thermal lost": "ThermalLost", "thermal lost (flagged)": "ThermalLostFlagged",
        "visible dark": "VisibleDark", "thermal noise": "ThermalNoise"}
COND_LABEL = {"Clean": "Clean", "ThermalLost": "Thermal lost", "ThermalLostFlagged": "Thermal lost, flagged",
              "VisibleDark": "Visible dark", "ThermalNoise": "Thermal noise"}
SUBSET = ["All", "Day", "Night", "Failed", "Healthy"]
TRACK_COND = {"clean": "Clean", "thermal lost": "ThermalLost"}


def sign_test_p(wins: int, n: int) -> float:
    """One-sided sign-test probability of at least `wins` successes in n fair trials."""
    return sum(math.comb(n, k) for k in range(wins, n + 1)) / 2 ** n


def sig_block(data) -> None:
    """Paired t-tests (df 2) and sign tests of v2 over concat, three seeds."""
    names = group("significance, v2 over concat with modality dropout, paired t-test (df 2, t 4.303), pp")
    for ds in DATASETS:
        for tag, key in (("", "mAP50-95(B)"), ("Small", "mAP_small(B)")):
            diffs = [data[(ds, "CffmVtwo", s)][key] - data[(ds, "ConcatMd", s)][key] for s in (0, 1, 2)]
            p, half, mean = ttest_df2(diffs)
            stem = f"sig{DATASETS[ds]}{tag}"
            put(names, stem + "P", q(p, 2))
            put(names, stem + "CiLow", signed(mean - half))
            put(names, stem + "CiHigh", signed(mean + half))
            wins = sum(d > 0 for d in diffs)
            put(names, stem + "Wins", str(wins))
            put(names, stem + "SignP", q(sign_test_p(wins, 3), 3))
            put(names, stem + "SignTwoSidedP", q(min(1.0, 2 * sign_test_p(max(wins, 3 - wins), 3)), 2))
    put(names, "sigSeedCount", "3")


def gain_block(data) -> None:
    """Fusion gain over the single-camera baselines on M3FD."""
    names = group("M3FD: fusion over single-camera YOLO26-n at 80 epochs, mAP in percentage points (single-camera rows are one seed)")
    vis, thm = (data[("m3fd", m, 0)]["mAP50-95(B)"] for m in ("YoloVis", "YoloThm"))
    best, best_name = (vis, "visible") if vis >= thm else (thm, "thermal")
    put(names, "gainMfdBestSingle", best_name)
    put(names, "gainMfdBestSingleMap", q(best, 1, 100))
    put(names, "gainMfdVisOverThmMap", signed(vis - thm))
    for model in ("CffmVtwo", "ConcatMd"):
        per = {s: data[("m3fd", model, s)]["mAP50-95(B)"] for s in (0, 1, 2)}
        for tag, ref in (("Best", best), ("Vis", vis), ("Thm", thm)):
            diffs = [v - ref for v in per.values()]
            mean, sd = spread(diffs)
            put(names, f"gainMfd{model}Over{tag}MeanMap", signed(mean))
            put(names, f"gainMfd{model}Over{tag}StdMap", q(sd, 1, 100))
            put(names, f"gainMfd{model}Over{tag}MinMap", signed(min(diffs)))
    for tag, key in (("MapFifty", "mAP50(B)"), ("ApSmall", "mAP_small(B)")):
        v, t = (data[("m3fd", m, 0)][key] for m in ("YoloVis", "YoloThm"))
        ref = max(v, t)
        diffs = [data[("m3fd", "CffmVtwo", s)][key] - ref for s in (0, 1, 2)]
        mean, sd = spread(diffs)
        put(names, f"gainMfdCffmVtwoOverBest{tag}Mean", signed(mean))
        put(names, f"gainMfdCffmVtwoOverBest{tag}Std", q(sd, 1, 100))


def latency_local_block() -> None:
    """Emit the re-timed latency on the local GPU."""
    lat = load(RES / "latency_local.json")
    names = group("latency re-timed on an idle local GPU, FP16, batch 1: parameters in millions, GFLOPs, milliseconds")
    put(names, "latLocalDevice", lat["device"].replace("NVIDIA GeForce ", ""))
    for run, tok in LAT_LOCAL.items():
        v = lat[run]
        put(names, f"latLocal{tok}ParamsM", q(v["params_M"], 2))
        put(names, f"latLocal{tok}Gflops", q(v["GFLOPs"], 1))
        put(names, f"latLocal{tok}MedianMs", q(v["median_ms"], 1))
        put(names, f"latLocal{tok}PNinetyFiveMs", q(v["p95_ms"], 1))
        put(names, f"latLocal{tok}Fps", q(1000 / v["median_ms"], 0))
    a, b, t = lat["cffm_v2"], lat["concat_md"], lat["yolo26n_thermal"]
    put(names, "latLocalRatioMedian", q(a["median_ms"] / b["median_ms"], 2))
    put(names, "latLocalRatioParams", q(a["params_M"] / b["params_M"], 2))
    put(names, "latLocalRatioGflops", q(a["GFLOPs"] / b["GFLOPs"], 2))
    put(names, "latLocalRatioPNinetyFive", q(a["p95_ms"] / b["p95_ms"], 2))
    put(names, "latLocalVtwoOverThmMedian", q(a["median_ms"] / t["median_ms"], 2))
    put(names, "latLocalVthreeOverVtwoMedian", q(lat["cffm_v3"]["median_ms"] / a["median_ms"], 2))


def phase_two_block() -> None:
    """Emit KAIST stills, memory and tracking macros."""
    summ = load(RES / "phase2" / "phase2_summary.json")
    for k, f in (("stills", "stills.json"), ("memory", "memory.json"), ("tracking", "tracking.json")):
        if k != "tracking" and load(RES / "phase2" / f) != summ[k]:
            CHECKS.append(f"DISAGREE phase2/{f} vs phase2_summary.json")
    stills = summ["stills"]
    names = group("KAIST stills, reasonable subset, log-average miss rate and AP50 in percent, test images as integers")
    for m, tok in (("concat", "Concat"), ("cffm", "Cffm")):
        for sub in ("all", "day", "night"):
            put(names, f"kst{tok}{sub.capitalize()}Mr", q(stills[m][f"MR_{sub}"], 1, 100))
            put(names, f"kst{tok}{sub.capitalize()}ApFifty", q(stills[m][f"AP50_{sub}"], 1, 100))
    for sub in ("all", "day", "night"):
        put(names, f"kstN{sub.capitalize()}", str(stills["cffm"][f"n_{sub}"]))
        put(names, f"kstMargin{sub.capitalize()}Mr", signed(stills["cffm"][f"MR_{sub}"] - stills["concat"][f"MR_{sub}"]))
        put(names, f"kstMargin{sub.capitalize()}ApFifty",
            signed(stills["cffm"][f"AP50_{sub}"] - stills["concat"][f"AP50_{sub}"]))
    mem = summ["memory"]
    names = group("Temporal State Memory on KAIST: miss rate in percent, memory change in points (negative is better)")
    for cond, ctok in COND.items():
        img, mm = mem[f"{cond} | image"], mem[f"{cond} | memory"]
        for sub in SUBSET:
            if f"MR_{sub.lower()}" not in img:
                continue
            if sub in ("Failed", "Healthy") and ctok == "Clean":
                continue
            for mode, d in (("Image", img), ("Memory", mm)):
                put(names, f"mem{ctok}{mode}{sub}Mr", q(d[f"MR_{sub.lower()}"], 1, 100))
            put(names, f"memChange{ctok}{sub}Mr", signed(mm[f"MR_{sub.lower()}"] - img[f"MR_{sub.lower()}"]))
        for mode, d in (("Image", img), ("Memory", mm)):
            put(names, f"mem{ctok}{mode}AllApFifty", q(d["AP50_all"], 1, 100))
        put(names, f"memChange{ctok}AllApFifty", signed(mm["AP50_all"] - img["AP50_all"]))
        if "n_failed" in img:
            put(names, f"memN{ctok}Failed", str(img["n_failed"]))
            put(names, f"memN{ctok}Healthy", str(img["n_healthy"]))
    put(names, "memNAll", str(mem["clean | image"]["n_all"]))
    trk = load(RES / "phase2" / "tracking.json")
    names = group("tracking on four KAIST test sequences with ByteTrack: MOTA and IDF1 in percent, switches as integers")
    put(names, "trkSeqCount", str(len(trk["clean | image"])))
    put(names, "trkUniqueTotal", str(int(sum(r["num_unique_objects"] for r in trk["clean | image"]))))
    for cond, ctok in TRACK_COND.items():
        per = {mode: trk[f"{cond} | {mode}"] for mode in ("image", "memory")}
        for mode, rows in per.items():
            mt = mode.capitalize()
            put(names, f"trk{ctok}{mt}Mota", q(statistics.mean(r["mota"] for r in rows), 1, 100))
            put(names, f"trk{ctok}{mt}IdfOne", q(statistics.mean(r["idf1"] for r in rows), 1, 100))
            put(names, f"trk{ctok}{mt}Switches", str(int(sum(r["num_switches"] for r in rows))))
            put(names, f"trk{ctok}{mt}MostlyTracked", str(int(sum(r["mostly_tracked"] for r in rows))))
        for key, tok in (("mota", "Mota"), ("idf1", "IdfOne")):
            d = [b[key] - a[key] for a, b in zip(per["image"], per["memory"])]
            put(names, f"trkChange{ctok}{tok}", signed(statistics.mean(d)))
            put(names, f"trk{ctok}MemoryWins{tok}", str(sum(x > 0 for x in d)))
    for mode in ("image", "memory"):
        for key, tok in (("mota", "Mota"), ("idf1", "IdfOne")):
            cost = (statistics.mean(r[key] for r in trk[f"clean | {mode}"])
                    - statistics.mean(r[key] for r in trk[f"thermal lost | {mode}"]))
            put(names, f"trkFaultCost{mode.capitalize()}{tok}", q(cost, 1, 100))
    flat = {k: {m: statistics.mean(r[m] for r in v) for m in ("mota", "idf1", "num_switches")} for k, v in trk.items()}
    for k, v in flat.items():
        for m, x in v.items():
            if abs(x - summ["tracking"][k][m]) > 1e-9:
                CHECKS.append(f"DISAGREE phase2 tracking summary {k} {m}")


def phase_three_block() -> None:
    """Emit live runtime macros."""
    summ = load(RES / "phase3" / "phase3_summary.json")
    if load(RES / "phase3" / "soak.json") != summ["soak"] or load(RES / "phase3" / "faults.json") != summ["faults"]:
        CHECKS.append("DISAGREE phase3 soak/faults.json vs phase3_summary.json")
    runs = {"SoakThirty": summ["soak"]["30 fps"], "SoakTwenty": summ["soak"]["20 fps"],
            "FaultOn": {**summ["faults"]["monitor on"]["summary"]},
            "FaultOff": {**summ["faults"]["monitor off"]["summary"]}}
    names = group("live runtime: seconds, throughput, delays in milliseconds, drift, GPU memory growth, capture drops")
    for tag, s in runs.items():
        stem = f"liv{tag}"
        put(names, stem + "Seconds", q(s["seconds"], 0))
        put(names, stem + "Fps", q(s["fps"], 1))
        put(names, stem + "EndToEndMedianMs", q(s["e2e_p50"], 1))
        put(names, stem + "EndToEndPNinetyFiveMs", q(s["e2e_p95"], 1))
        put(names, stem + "InferMedianMs", q(s["infer_p50"], 1))
        put(names, stem + "InferPNinetyFiveMs", q(s["infer_p95"], 1))
        put(names, stem + "DriftMs", signed(s["drift_ms"], 1, 1))
        put(names, stem + "GpuGrowthMb", signed(s["gpu_growth_mb"], 0, 1))
        if "capture_drop_rate" in s:
            put(names, stem + "DropPercent", q(s["capture_drop_rate"], 1, 100))
    put(names, "livSoakThirtyNominalFps", "30")
    put(names, "livSoakTwentyNominalFps", "20")
    names = group("live runtime with scripted link faults: monitor on and off")
    for tag, key in (("On", "monitor on"), ("Off", "monitor off")):
        d = summ["faults"][key]
        put(names, f"livFault{tag}Mr", q(d["MR_all"], 1, 100))
        put(names, f"livFault{tag}Reconnects", str(d["reconnects"]))
        for fault, tok in (("thermal unplugged", "Unplugged"), ("visible frozen", "Frozen"), ("thermal stalled", "Stalled")):
            f = d["faults"][fault]
            put(names, f"livFault{tag}{tok}DetectS", q(f["detect_s"], 2))
            put(names, f"livFault{tag}{tok}RecoverS", q(f["recover_s"], 2))
            put(names, f"livFault{tag}{tok}FalseAlarmPercent", q(f["false_alarm_rate"], 1, 100))
    put(names, "livFaultMrChange", signed(summ["faults"]["monitor on"]["MR_all"] - summ["faults"]["monitor off"]["MR_all"]))


def sota_block() -> None:
    """Emit ranges of the published numbers quoted in the prose."""
    s = load(Path(__file__).resolve().parent / "sota.json")
    names = group("published results copied from the cited papers: ranges quoted in the text")
    fusion = [r for r in s["llvip"] if "infrared only" not in r["method"]]
    maps = [float(r["map"]) for r in fusion]
    put(names, "sotaLlvipFusionMapMin", q(min(maps), 1))
    put(names, "sotaLlvipFusionMapMax", q(max(maps), 1))
    put(names, "sotaLlvipIrOnlyMap", q(float(next(r for r in s["llvip"] if "infrared only" in r["method"])["map"]), 1))
    m50 = [float(r["map50"]) for r in s["m3fd"]]
    put(names, "sotaMfdMapFiftyMin", q(min(m50), 1))
    put(names, "sotaMfdMapFiftyMax", q(max(m50), 1))
    put(names, "sotaMfdSplitSwing", q(float(s["extra"]["cft_split_high"]) - float(s["extra"]["cft_split_low"]), 1))
    mr = [float(r["all"]) for r in s["kaist"]]
    put(names, "sotaKaistMrMin", q(min(mr), 2))
    put(names, "sotaKaistMrMax", q(max(mr), 2))


def sota_tables() -> None:
    """Write the published-results context tables from tools/sota.json plus this paper's rows."""
    s = load(Path(__file__).resolve().parent / "sota.json")
    ours = {"llvip": [("CFFM-Net v2 (this work)", f"YOLO26-n, {BS}latLocalVtwoParamsM{{}}{BS},M", f"{BS}cfgImgsz{{}}, every third frame", "resLlvipCffmVtwoMeanMapFifty",
                       "resLlvipCffmVtwoMeanMap"),
                      ("Concat baseline (this work)", f"YOLO26-n, {BS}latLocalConcatMdParamsM{{}}{BS},M", f"{BS}cfgImgsz{{}}, every third frame",
                       "resLlvipConcatMdMeanMapFifty", "resLlvipConcatMdMeanMap")],
            "m3fd": [("CFFM-Net v2 (this work)", f"YOLO26-n, {BS}latLocalVtwoParamsM{{}}{BS},M", "3,360 / 840, random", "resMfdCffmVtwoMeanMapFifty",
                      "resMfdCffmVtwoMeanMap"),
                     ("Concat baseline (this work)", f"YOLO26-n, {BS}latLocalConcatMdParamsM{{}}{BS},M", "3,360 / 840, random",
                      "resMfdConcatMdMeanMapFifty", "resMfdConcatMdMeanMap")]}
    body = [sub("LLVIP, official split (12,025 / 3,463 pairs)", 5)]
    for r in s["llvip"]:
        body.append(f"{r['method']}~\\cite{{{r['cite']}}} & {r['detector']} & {r['setting']} & {r['map50']} & {r['map']}")
    for name, det, setting, a, b in ours["llvip"]:
        body.append(f"{name} & {det} & {setting} & {cell(a)} & {cell(b)}")
    body += ["-", sub("M3FD, splits differ between rows and are not comparable", 5)]
    for r in s["m3fd"]:
        body.append(f"{r['method']}~\\cite{{{r['cite']}}} & {r['detector']} & {r['setting']} & {r['map50']} & {r['map']}")
    for name, det, setting, a, b in ours["m3fd"]:
        body.append(f"{name} & {det} & {setting} & {cell(a)} & {cell(b)}")
    table("tab_sota", "Published RGB-T detection results for context, in percent. Numbers are copied from the cited papers, "
          "not re-run. Input size or split is stated per row because mAP depends on both. Our models are nano scale, "
          "trained on one third of the LLVIP training frames (mean of three seeds), so they are not a like-for-like "
          "comparison. On M3FD no row shares a split with ours, and no ranking is claimed.", "tab:sota", "lllcc",
          "Method & Detector & Input or split & mAP$_{50}$ & mAP", body, star=True)
    body = []
    for r in s["kaist"]:
        body.append(f"{r['method']}~\\cite{{{r['cite']}}} & {r['detector']} & {r['train']} & {r['all']} & {r['day']} & {r['night']}")
    body += ["-"]
    for name, k in (("Concat baseline (this work)", "Concat"), ("CFFM-Net v2 (this work)", "Cffm")):
        body.append(f"{name} & YOLO26-n & original & " + " & ".join(cell(f"kst{k}{t}Mr") for t in ("All", "Day", "Night")))
    table("tab_sota_kaist", "KAIST log-average miss rate in percent on the reasonable subset (lower is better). Published rows "
          "use the improved test annotations and the standard evaluation code and are copied from the cited papers. "
          "Our rows use the original annotations for training and testing, every 20th test frame and our own implementation "
          "of the protocol, so they are indicative only.", "tab:sotakaist", "lllccc",
          "Method & Detector & Train annotations & All & Day & Night", body, star=True)


def kaist_table() -> None:
    """Write the KAIST still-image table."""
    rows = [("Miss rate, all", "All", "Mr"), ("Miss rate, day", "Day", "Mr"), ("Miss rate, night", "Night", "Mr")]
    body = [f"{lab} (\\%) & {cell(f'kstConcat{s}Mr')} & {cell(f'kstCffm{s}Mr')} & {cell(f'kstMargin{s}Mr')}" for lab, s, _ in rows]
    body.append(f"AP$_{{50}}$, all (\\%) & {cell('kstConcatAllApFifty')} & {cell('kstCffmAllApFifty')} & {cell('kstMarginAllApFifty')}")
    table("tab_kaist", "KAIST test results on single frames, reasonable subset. Lower is better for the miss rate. One training "
          "run per model, \\kstNAll{} test frames, so no confidence interval is available.", "tab:kaist", "lccc",
          "Measure & Concat & CFFM-Net v2 & Change", body)


def memory_table() -> None:
    """Write the Temporal State Memory table."""
    head = (f"& {BS}multicolumn{{3}}{{c}}{{All frames}} & {BS}multicolumn{{3}}{{c}}{{Failed frames}} {BS}{BS} "
            f"{BS}cmidrule(lr){{2-4}}{BS}cmidrule(lr){{5-7}} Condition & Image & Memory & Change & Image & Memory & Change")
    body = [sub("Miss rate (\\%), lower is better", 7)]
    for c in COND.values():
        cells = []
        for sub_ in ("All", "Failed"):
            if f"mem{c}Image{sub_}Mr" in M:
                cells += [cell(f"mem{c}Image{sub_}Mr"), cell(f"mem{c}Memory{sub_}Mr"), cell(f"memChange{c}{sub_}Mr")]
            else:
                cells += ["--", "--", "--"]
        body.append(f"{COND_LABEL[c]} & " + " & ".join(cells))
    body += ["-", f"{BS}multicolumn{{7}}{{l}}{{{BS}emph{{Tracking, mean of four sequences}}}} {BS}{BS} "
             f"& {BS}multicolumn{{3}}{{c}}{{MOTA (\\%)}} & {BS}multicolumn{{3}}{{c}}{{IDF1 (\\%)}} {BS}{BS} "
             f"{BS}cmidrule(lr){{2-4}}{BS}cmidrule(lr){{5-7}} Condition & Image & Memory & Change & Image & Memory & Change"]
    for c in TRACK_COND.values():
        cells = []
        for t in ("Mota", "IdfOne"):
            cells += [cell(f"trk{c}Image{t}"), cell(f"trk{c}Memory{t}"), cell(f"trkChange{c}{t}")]
        body.append(f"{COND_LABEL[c]} & " + " & ".join(cells))
    table("tab_memory", "Temporal State Memory against the same detector without memory on KAIST. A camera fails for four "
          "seconds in every eight. Failed frames are those inside a failure. Change is memory minus image. One memory "
          "training run, one detector.", "tab:memory", "lcccccc", head, body, star=True)


def live_table() -> None:
    """Write the live runtime tables."""
    cols = ["SoakThirty", "SoakTwenty", "FaultOn", "FaultOff"]
    rows = [("Run time (s)", "Seconds"), ("Throughput (frames/s)", "Fps"), ("Capture drop (\\%)", "DropPercent"),
            ("End-to-end median (ms)", "EndToEndMedianMs"), ("End-to-end p95 (ms)", "EndToEndPNinetyFiveMs"),
            ("Inference median (ms)", "InferMedianMs"), ("Inference p95 (ms)", "InferPNinetyFiveMs"),
            ("Drift (ms)", "DriftMs"), ("GPU memory growth (MB)", "GpuGrowthMb")]
    head = (f"& {BS}multicolumn{{2}}{{c}}{{Soak}} & {BS}multicolumn{{2}}{{c}}{{Link faults}} {BS}{BS} "
            f"{BS}cmidrule(lr){{2-3}}{BS}cmidrule(lr){{4-5}} Measure & 30 fps & 20 fps & Monitor on & Monitor off")
    body = [f"{lab} & " + " & ".join(cell(f"liv{c}{k}") for c in cols) for lab, k in rows]
    body.append(f"Miss rate, all frames (\\%) & -- & -- & {cell('livFaultOnMr')} & {cell('livFaultOffMr')}")
    body.append(f"Camera reconnects & -- & -- & {cell('livFaultOnReconnects')} & {cell('livFaultOffReconnects')}")
    table("tab_live", "Live runtime on replayed KAIST sequences. Soak: paired cameras at the stated rate. Link faults: "
          "20\\,fps with a scripted cable pull, freeze and stall. Drift is the late minus the early median end-to-end "
          "delay. One run per row.", "tab:live", "lcccc", head, body)
    body = []
    for lab, tok in (("Thermal unplugged, 6\\,s", "Unplugged"), ("Visible frozen, 6\\,s", "Frozen"), ("Thermal stalled, 3\\,s", "Stalled")):
        body.append(f"{lab} & " + " & ".join(cell(f"livFault{m}{tok}{k}") for k in ("DetectS", "RecoverS", "FalseAlarmPercent")
                                              for m in ("On", "Off")))
    head = (f"& {BS}multicolumn{{2}}{{c}}{{Detect (s)}} & {BS}multicolumn{{2}}{{c}}{{Recover (s)}} & "
            f"{BS}multicolumn{{2}}{{c}}{{False alarms (\\%)}} {BS}{BS} "
            f"{BS}cmidrule(lr){{2-3}}{BS}cmidrule(lr){{4-5}}{BS}cmidrule(lr){{6-7}} Fault & On & Off & On & Off & On & Off")
    table("tab_monitor", "Stream monitor timing for each scripted fault, with the monitor feeding the model (on) and "
          "only observing (off). Delays are seconds from onset or end of the fault to the first change of state.",
          "tab:monitor", "lcccccc", head, body)


def main() -> None:
    """Build every output and print the cross-check report."""
    data = results_block()
    sig_block(data)
    gain_block(data)
    faults_block()
    latency_block()
    latency_local_block()
    probe_block()
    hyp_block()
    phase_block()
    phase_two_block()
    phase_three_block()
    sota_block()
    cross_checks(data)
    write_numbers()
    (SRC / "tables").mkdir(exist_ok=True)
    main_tables()
    fault_table()
    probe_table()
    latency_table()
    camera_table()
    kaist_table()
    memory_table()
    live_table()
    sota_tables()
    print(f"{len(M)} macros")
    for c in sorted(set(CHECKS)):
        print(c)


if __name__ == "__main__":
    sys.exit(main())
