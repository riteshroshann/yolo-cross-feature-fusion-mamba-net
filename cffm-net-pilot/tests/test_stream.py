import time

import cv2
import numpy as np
import pytest
import torch

from cffm import stream
from cffm.model import DualStreamDetectionModel

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def test_latest_keeps_only_newest():
    q = stream.Latest()
    for i in range(5):
        q.put(stream.Frame(np.zeros(1), float(i), i))
    assert q.take().seq == 4 and q.dropped == 4 and q.take(timeout=0) is None


@pytest.fixture(scope="module")
def frames(tmp_path_factory):
    d = tmp_path_factory.mktemp("cam")
    rng = np.random.default_rng(0)
    paths = []
    for i in range(30):
        p = d / f"{i:03d}.jpg"
        cv2.imwrite(str(p), rng.integers(20, 200, (128, 160, 3), dtype=np.uint8))
        paths.append(p)
    return paths


def test_monitor_flags_unplug_and_recovers(frames):
    m = DualStreamDetectionModel("configs/models/cffm-net-n-v2.yaml", nc=1, verbose=False).to(DEV)
    m.attach_memory()
    t0 = time.perf_counter() + 0.3
    vis = stream.Camera("visible", lambda: stream.ReplaySource(frames, fps=20, t0=t0))
    ir = stream.Camera("thermal", lambda: stream.ReplaySource(
        frames, fps=20, grey=True, t0=t0, faults=stream.StreamFaults(unplugs=[(1.5, 2.5)])), grey=True, reconnect=0.05)
    rt = stream.Runtime(m, vis, ir, fps=20, imgsz=(128, 160), device=DEV)
    rec = rt.run(4.5)
    rt.close()
    s = stream.summary(rec, skip_s=0.5)
    assert s["fps"] > 5 and abs(s["drift_ms"]) < 200
    d = stream.detection_delay(rec, 1.5, 2.5, camera="ir", start=t0)
    assert d["detect_s"] < 0.6 and d["recover_s"] < 1.0 and d["false_alarm_rate"] < 0.05
    assert any(r["ir"] == "down" for r in rec) and rec[-1]["ir"] == "ok"
    assert ir.reconnects >= 1


def test_monitor_detects_freeze_and_black():
    mon = stream.Monitor(nominal_fps=20, freeze_frames=3)
    img = np.full((24, 32), 100, np.uint8)
    now = 0.0
    for i in range(5):
        now += 0.05
        mon.observe("c", stream.Frame(img, now, i), now)
    assert mon.health("c", stream.Frame(img, now, 5), now).frozen
    black = stream.Frame(np.zeros((24, 32), np.uint8), now, 6)
    assert "black" in mon.health("c", black, now).reasons
