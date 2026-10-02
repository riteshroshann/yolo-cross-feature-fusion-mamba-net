"""End-to-end wiring check: train one epoch on a fake dataset, evaluate, probe.

Not a result, a plumbing test. It proves that the Ultralytics trainer,
DataLoader workers (spawned on Windows), validator, size-binned COCO evaluation
and the degradation probe all work with our model and data. Slow-ish (about a
minute), so it only runs when CFFM_SMOKE=1:

    CFFM_SMOKE=1 python -m pytest tests/test_smoke_train.py -q
"""
import os
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.skipif(os.environ.get("CFFM_SMOKE") != "1", reason="set CFFM_SMOKE=1 to run")
ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def synth(tmp_path_factory):
    from cffm.data import make_synthetic

    return make_synthetic(tmp_path_factory.mktemp("smoke") / "synthetic", n_train=8, n_val=4, imgsz=320)


@pytest.mark.parametrize("model,data", [
    ("configs/models/cffm-net-n.yaml", "data_paired.yaml"),
    ("configs/models/two-stream-concat-n.yaml", "data_paired.yaml"),
    ("yolo26n.yaml", "data_visible.yaml"),
])
def test_one_epoch_then_evaluate(synth, tmp_path, model, data):
    from cffm.train import evaluate, train_run

    device = 0 if torch.cuda.is_available() else "cpu"
    best = train_run(f"smoke_{Path(model).stem}", str(ROOT / model) if model.startswith("configs") else model,
                     str(synth / data), pretrained=str(ROOT / "weights" / "yolo26n.pt"), epochs=1, imgsz=320,
                     batch=4, device=device, project=str(tmp_path), workers=2, amp=False, plots=False)
    assert best.exists()
    m = evaluate(best, synth / data, imgsz=320, batch=4, device=device, project=str(tmp_path / "eval"))
    for k in ("mAP50(B)", "mAP50-95(B)", "AP_vt(B)", "AP_s(B)"):
        assert k in m, f"missing {k}: {sorted(m)}"


def test_probe_runs_on_a_paired_checkpoint(synth, tmp_path):
    from cffm.train import probe, train_run

    device = 0 if torch.cuda.is_available() else "cpu"
    best = train_run("smoke_probe", str(ROOT / "configs/models/cffm-net-n.yaml"), str(synth / "data_paired.yaml"),
                     pretrained=str(ROOT / "weights" / "yolo26n.pt"), epochs=1, imgsz=320, batch=4, device=device,
                     project=str(tmp_path), workers=0, amp=False, plots=False)
    for kind in ("clean", "thermal_drop", "thermal_shift_8"):
        m = probe(best, synth / "data_paired.yaml", kind, imgsz=320, batch=4, device=device,
                  project=str(tmp_path / "probe"))
        assert "mAP50-95(B)" in m
    m = probe(best, synth / "data_paired.yaml", "thermal_drop", flagged=True, imgsz=320, batch=4, device=device,
              project=str(tmp_path / "probe"))
    assert "mAP50-95(B)" in m
