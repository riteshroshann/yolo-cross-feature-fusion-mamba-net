"""Run the sensor-failure benchmark over trained checkpoints; resumable, results in results/faults.csv."""
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cffm import faults
from cffm.train import fault_eval

OUT = ROOT / "results" / "faults.csv"
FIELDS = ["run", "dataset", "fault", "severity", "flagged", "AP", "AP50", "AP_s"]


def cases():
    yield "clean", 0, False
    for f in faults.FAULTS:
        for s in faults.SEVERITIES:
            yield f, s, False
        if faults.flags_for(f):
            yield f, 3, True


def main(patterns):
    done = set()
    if OUT.exists():
        done = {(r["run"], r["fault"], r["severity"], r["flagged"]) for r in csv.DictReader(open(OUT))}
    new = not OUT.exists()
    with open(OUT, "a", newline="") as fh:
        w = csv.DictWriter(fh, FIELDS)
        if new:
            w.writeheader()
        for pat in patterns:
            for wt in sorted(ROOT.glob(f"runs/{pat}/weights/best.pt")):
                run = wt.parent.parent.name
                ds = run.split("_")[0]
                data = ROOT / "data" / "processed" / ds / "data_paired.yaml"
                for fault, sev, flagged in cases():
                    key = (run, fault, str(sev), str(flagged))
                    if key in done:
                        continue
                    m = fault_eval(wt, data, fault, max(sev, 1), flagged=flagged, batch=8, device=0)
                    w.writerow(dict(run=run, dataset=ds, fault=fault, severity=sev, flagged=flagged,
                                    AP=round(m["mAP50-95(B)"], 4), AP50=round(m["mAP50(B)"], 4),
                                    AP_s=round(m.get("AP_s(B)", float("nan")), 4)))
                    fh.flush()
                    print(run, fault, sev, "flagged" if flagged else "", round(m["mAP50-95(B)"], 4), flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or ["*_e80"])
