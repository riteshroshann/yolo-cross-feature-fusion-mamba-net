"""Plot the per-fault mAP margin of CFFM-Net v2 over concat at severity three, from numbers.tex."""
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

HERE = Path(__file__).resolve().parent.parent
FAULTS = ["VisibleDark", "VisibleGlare", "VisibleFog", "VisibleNoise", "VisibleBlur", "VisibleDrop",
          "ThermalDrop", "ThermalShift", "ThermalNoise", "ThermalCrossover", "ThermalDead"]
LABELS = ["dark", "glare", "fog", "noise", "blur", "drop", "drop", "shift", "noise", "crossover", "dead"]
VIS, THM = "#2F6DB5", "#C0622B"


def load() -> dict:
    text = (HERE / "numbers.tex").read_text(encoding="utf-8")
    return dict(re.findall(r"\\newcommand\{\\(\w+)\}\{(.*)\}\s*$", text, re.M))


def num(s: str) -> float:
    return float(s.replace("\\ensuremath{-}", "-").replace("\\ensuremath{+}", "+"))


def margins(nums: dict, key: str) -> tuple[list[float], list[float] | None]:
    """Margins per fault; for several seeds also the sample std over seeds."""
    one = [f"flt{key}Margin{f}SevThreeMap" for f in FAULTS]
    if all(k in nums for k in one):
        return [num(nums[k]) for k in one], None
    vals = [num(nums[f"flt{key}Margin{f}SevThreeMeanMap"]) for f in FAULTS]
    errs = [num(nums[f"flt{key}Margin{f}SevThreeStdMap"]) for f in FAULTS]
    return vals, errs


def main() -> None:
    nums = load()
    plt.rcParams.update({"font.family": "serif", "font.size": 8,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.5))
    for ax, (name, key) in zip(axes, (("LLVIP, seed 0", "Llvip"), ("M3FD, mean of three seeds", "Mfd"))):
        vals, errs = margins(nums, key)
        ax.bar(range(len(vals)), vals, color=[VIS if f.startswith("Visible") else THM for f in FAULTS], width=0.7,
               yerr=errs, error_kw={"lw": 0.7, "capsize": 1.5, "ecolor": "#333"})
        ax.axhline(0, color="#444", lw=0.6)
        ax.set_xticks(range(len(vals)))
        ax.set_xticklabels(LABELS, rotation=45, ha="right")
        ax.set_title(name)
        ax.set_ylabel("v2 minus concat, mAP (pp)")
    axes[0].legend(handles=[Patch(color=VIS, label="visible fault"), Patch(color=THM, label="thermal fault")],
                   frameon=False, loc="lower left")
    fig.tight_layout()
    fig.savefig(HERE / "figures" / "fig_fault_margin.pdf")


if __name__ == "__main__":
    sys.exit(main())
