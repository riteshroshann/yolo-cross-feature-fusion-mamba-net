from pathlib import Path

import numpy as np
import pytest

from cffm import kaist

VBB = Path(__file__).resolve().parents[1] / "data" / "raw" / "kaist_probe" / "annotations" / "set06" / "V000.vbb"


def _b(x, y, w, h, label="person", vis=1.0, tid=0):
    return kaist.Box(tid, label, (x, y, w, h), vis)


@pytest.mark.skipif(not VBB.exists(), reason="KAIST probe annotations not downloaded")
def test_read_vbb():
    frames = kaist.read_vbb(VBB)
    assert len(frames) == 1523
    boxes = [b for f in frames for b in f]
    assert boxes and all(b.xywh[2] > 0 and b.xywh[3] > 0 and 0 <= b.vis <= 1 for b in boxes)
    assert {b.label for b in boxes} <= {"person", "people", "cyclist", "person?", "person?a"}


def test_reasonable_subset():
    assert _b(0, 0, 30, 60).reasonable
    assert not _b(0, 0, 30, 50).reasonable
    assert not _b(0, 0, 30, 60, vis=0.3).reasonable
    assert not _b(0, 0, 30, 60, label="people").reasonable


def test_miss_rate_perfect_and_empty():
    gts = [[_b(10, 10, 30, 80)], [_b(100, 50, 40, 90)]]
    perfect = [np.array([[10, 10, 40, 90, 0.9, 0]]), np.array([[100, 50, 140, 140, 0.8, 0]])]
    assert kaist.miss_rate(perfect, gts) < 1e-6
    assert kaist.miss_rate([np.zeros((0, 6)), np.zeros((0, 6))], gts) == pytest.approx(1.0)


def test_miss_rate_ignores_non_reasonable_and_penalises_fp():
    gts = [[_b(10, 10, 30, 80), _b(200, 10, 20, 30, label="people")]]
    hit_ignored = [np.array([[10, 10, 40, 90, 0.9, 0], [200, 10, 220, 40, 0.95, 0]])]
    assert kaist.miss_rate(hit_ignored, gts) < 1e-6
    many_fp = [np.array([[10, 10, 40, 90, 0.5, 0]] + [[300 + i, 0, 340 + i, 80, 0.9, 0] for i in range(0, 200, 50)])]
    assert kaist.miss_rate(many_fp, gts) > 0.3
