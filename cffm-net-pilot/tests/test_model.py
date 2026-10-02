"""The dual-stream model builds from every config, runs, and loads COCO weights correctly."""
from pathlib import Path

import pytest
import torch

from cffm.blocks import CMFM, ConcatFusion, ReliabilityWeightedSum
from cffm.model import DualStreamDetectionModel

ROOT = Path(__file__).resolve().parents[1]
CFGS = sorted((ROOT / "configs" / "models").glob("*.yaml"))
WEIGHTS = ROOT / "weights" / "yolo26n.pt"


@pytest.mark.parametrize("cfg", CFGS, ids=lambda p: p.stem)
def test_builds_and_runs(cfg):
    m = DualStreamDetectionModel(str(cfg), nc=3, verbose=False)
    x = torch.rand(2, 4, 128, 160)
    m.train()
    out = m(x)
    assert isinstance(out, dict) and "one2many" in out
    m.eval()
    with torch.no_grad():
        y = m(x)
    preds = y[0] if isinstance(y, (tuple, list)) else y
    assert preds.shape[0] == 2 and preds.shape[1] == 4 + 3


def test_fusion_layout_of_cffm_net():
    m = DualStreamDetectionModel(str(ROOT / "configs/models/cffm-net-n.yaml"), nc=1, verbose=False)
    assert m.fuse_idx == [2, 4, 6, 10]
    assert [m.levels[i] for i in m.fuse_idx] == [2, 3, 4, 5]
    assert isinstance(m.fusers["2"], ReliabilityWeightedSum)
    assert all(isinstance(m.fusers[k], CMFM) for k in ("4", "6", "10"))


def test_baseline_has_no_p2_and_concat_fusers():
    m = DualStreamDetectionModel(str(ROOT / "configs/models/two-stream-concat-n.yaml"), nc=1, verbose=False)
    assert m.fuse_idx == [4, 6, 10]
    assert all(isinstance(f, ConcatFusion) for f in m.fusers.values())


def test_three_channel_input_is_accepted():
    m = DualStreamDetectionModel(str(ROOT / "configs/models/cffm-net-n.yaml"), nc=1, verbose=False).eval()
    with torch.no_grad():
        m(torch.rand(1, 3, 128, 128))


@pytest.mark.skipif(not WEIGHTS.exists(), reason="weights/yolo26n.pt not downloaded")
def test_coco_weights_fill_both_backbones():
    m = DualStreamDetectionModel(str(ROOT / "configs/models/cffm-net-n.yaml"), nc=1, verbose=False)
    m.load(torch.load(WEIGHTS, map_location="cpu", weights_only=False))
    v, t = m.model[: m.nb + 1].state_dict(), m.thermal.state_dict()
    assert v.keys() == t.keys() and all(torch.equal(v[k], t[k]) for k in v)
