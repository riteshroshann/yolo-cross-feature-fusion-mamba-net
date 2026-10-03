"""The code that decides the pilot hypotheses gets tested like any other code."""
import math

from cffm import report

EV = {
    "llvip_yolo26n_thermal": {"mAP50-95(B)": 0.600},
    "llvip_concat": {"mAP50-95(B)": 0.610},
    "llvip_cffm": {"mAP50-95(B)": 0.630},
    "llvip_cffm_nogate": {"mAP50-95(B)": 0.625},
    "llvip_cffm_gconv": {"mAP50-95(B)": 0.640},
    "m3fd_concat": {"mAP50-95(B)": 0.50, "AP_t(B)": 0.20, "AP_s(B)": 0.30},
    "m3fd_cffm": {"mAP50-95(B)": 0.52, "AP_t(B)": 0.22, "AP_s(B)": 0.32},
}
PR = {
    "llvip_cffm_clean": {"mAP50-95(B)": 0.63}, "llvip_cffm_thermal_drop": {"mAP50-95(B)": 0.40},
    "llvip_cffm_nogate_clean": {"mAP50-95(B)": 0.625}, "llvip_cffm_nogate_thermal_drop": {"mAP50-95(B)": 0.30},
}
LAT = {"llvip_cffm": {"median_ms": 9.0}, "llvip_concat": {"median_ms": 5.0}}


def test_verdict_bands():
    assert report.verdict(0.02, 0.015) == "supported"
    assert report.verdict(-0.02, 0.015) == "against"
    assert report.verdict(0.01, 0.015) == "inconclusive"
    assert report.verdict(math.nan, 0.015) == "missing run"


def test_degradation_advantage():
    adv = report.mean_degradation_advantage(PR, "llvip_cffm", "llvip_cffm_nogate", ["clean", "thermal_drop"])
    assert abs(adv - 0.095) < 1e-9


def test_decide():
    h = report.decide(EV, PR, LAT, ["clean", "thermal_drop"])
    assert h.loc["PH1", "verdict"] == "supported"
    assert h.loc["PH2", "verdict"] == "supported"
    assert h.loc["PH3", "verdict"] == "supported"
    assert h.loc["PH4", "verdict"] == "against"
    assert h.loc["PH5", "verdict"] == "supported"


def test_probe_table_does_not_confuse_prefixes():
    t = report.probe_table(PR, ["llvip_cffm", "llvip_cffm_nogate"], ["clean", "thermal_drop"])
    assert t.loc["thermal_drop", "llvip_cffm"] == 0.40 and t.loc["thermal_drop", "llvip_cffm_nogate"] == 0.30


def test_writers():
    t = report.main_table(EV, LAT)
    md = report.to_markdown(t)
    tex = report.to_latex(t, "Pilot results.", "tab:pilot")
    assert md.startswith("| run |") and "llvip_cffm" in md
    assert "\\begin{tabular}" in tex and "llvip\\_cffm" in tex and "\\hline" not in tex
