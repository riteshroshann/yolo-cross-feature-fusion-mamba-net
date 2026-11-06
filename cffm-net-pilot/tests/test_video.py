import cv2
import numpy as np
import pytest
import torch

from cffm import kaist, video
from cffm.model import DualStreamDetectionModel

DEV = "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture(scope="module")
def seq(tmp_path_factory):
    root = tmp_path_factory.mktemp("kaist") / "set06" / "V000"
    rng = np.random.default_rng(0)
    s = kaist.Sequence("set06/V000")
    for i in range(24):
        vis = rng.integers(0, 60, (128, 160, 3), dtype=np.uint8)
        ir = rng.integers(0, 60, (128, 160), dtype=np.uint8)
        x = 20 + 3 * i
        vis[30:100, x:x + 30] = 200
        ir[30:100, x:x + 30] = 250
        for mod, im in (("visible", vis), ("lwir", ir)):
            (root / mod).mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(root / mod / f"I{i:05d}.jpg"), im)
        s.frames.append(root / "visible" / f"I{i:05d}.jpg")
        s.boxes.append([kaist.Box(0, "person", (x, 30, 30, 70), 1.0)])
    return s


@pytest.fixture(scope="module")
def model():
    m = DualStreamDetectionModel("configs/models/cffm-net-n-v2.yaml", nc=1, verbose=False).to(DEV)
    m.attach_memory()
    return m


def test_clip_dataset(seq):
    ds = video.ClipDataset([seq], T=4, stride=2, fault_p=1.0)
    clips, labels, fault = ds[0]
    assert clips.shape == (4, 128, 160, 4) and labels.shape[1] == 6
    assert set(labels[:, 0].long().tolist()) == {0, 1, 2, 3}
    assert fault is not None and 1 <= fault[2] < 4


def test_train_memory_learns_and_changes_only_memory(seq, model):
    before = {k: v.clone() for k, v in model.state_dict().items() if not k.startswith("memory.")}
    hist = video.train_memory(model, [seq] * 8, epochs=1, T=4, stride=2, batch=2, workers=0, device=DEV,
                              amp=DEV == "cuda", log=lambda *a: None)
    after = model.state_dict()
    assert all(torch.equal(v, after[k]) for k, v in before.items())
    assert any(p.abs().sum() > 0 for n, p in model.memory.named_parameters() if n.endswith("out.weight"))


def test_sequence_eval_and_tracking(seq, model):
    sched = video.periodic("thermal_drop", 3, on=8, off=8)
    dets, failed = video.run_sequence(model, seq, stride=2, schedule=sched, eval_every=4, device=DEV, flags=True)
    assert set(dets) == {0, 4, 8, 12, 16, 20} and any(failed.values()) and not all(failed.values())
    res = video.evaluate_sequences(model, [seq], stride=2, eval_every=4, device=DEV, schedule=sched)
    assert {"MR_all", "AP50_all", "MR_failed", "MR_healthy"} <= set(res)
    assert 0 <= res["MR_all"] <= 1
    tracks = video.track_sequence(model, seq, device=DEV, conf=0.0001)
    m = video.mot_metrics(tracks, seq)
    assert {"mota", "idf1", "num_switches"} <= set(m)
