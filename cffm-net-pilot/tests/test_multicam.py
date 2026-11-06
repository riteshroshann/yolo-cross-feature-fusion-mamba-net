from pathlib import Path

import numpy as np
import pytest

from cffm import multicam as mc

ROOT = Path(__file__).resolve().parents[1] / "data" / "raw" / "wildtrack_probe"
HAVE = (ROOT / "calibrations").exists() and (ROOT / "annotations_positions" / "00000000.json").exists()


@pytest.mark.skipif(not HAVE, reason="WILDTRACK probe not downloaded")
def test_calibration_projects_ground_truth_onto_boxes():
    views = mc.load_views(ROOT)
    errs = []
    for p in mc.load_annotations(ROOT, 0):
        xy = mc.position_to_cm([p["positionID"]])
        for v in p["views"]:
            if v["xmin"] < 0:
                continue
            uv = views[v["viewNum"]].to_image(xy)[0]
            foot = np.array([(v["xmin"] + v["xmax"]) / 2, v["ymax"]])
            errs.append(np.linalg.norm(uv - foot) / max(v["ymax"] - v["ymin"], 1))
    assert len(errs) > 50 and np.median(errs) < 0.1


@pytest.mark.skipif(not HAVE, reason="WILDTRACK probe not downloaded")
def test_ground_truth_boxes_fuse_back_to_positions():
    views = mc.load_views(ROOT)
    pts, sc = [[] for _ in views], [[] for _ in views]
    for p in mc.load_annotations(ROOT, 0):
        for v in p["views"]:
            if v["xmin"] >= 0:
                pts[v["viewNum"]].append([v["xmin"], v["ymin"], v["xmax"], v["ymax"], 0.9, 0])
    per = [mc.foot_points(views[i], np.array(pts[i])) for i in range(len(views))]
    fused, _ = mc.fuse([p for p, _ in per], [s for _, s in per], radius=60)
    res = mc.moda([fused], [mc.gt_positions(ROOT, 0)])
    assert res["MODA"] > 0.8 and res["recall"] > 0.9


def test_moda_and_tracking_and_analytics():
    gt = [np.array([[0, 0], [100, 100]]), np.array([[10, 0], [110, 100]])]
    assert mc.moda(gt, gt)["MODA"] == pytest.approx(1.0)
    assert mc.moda([np.zeros((0, 2))] * 2, gt)["MODA"] == pytest.approx(0.0)
    trk = mc.GroundTracker(gate=60, min_hits=1)
    for t in range(6):
        trk.update(np.array([[0, -100 + 50 * t]]), t)
    tracks = trk.all_tracks()
    assert len(tracks) == 1
    assert mc.line_crossings(tracks, 0) == {"forward": 1, "backward": 0}
    z = [mc.Zone("all", (-1e4, 1e4), (-1e4, 1e4))]
    assert mc.dwell_seconds(tracks, z, fps=1)["all"]["total_s"] == pytest.approx(5)


def test_anonymize_blurs_only_heads():
    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, (200, 200, 3), dtype=np.uint8)
    out = mc.anonymize(img, np.array([[50, 20, 110, 180]]))
    assert not np.array_equal(out[20:50, 50:110], img[20:50, 50:110])
    assert np.array_equal(out[80:180], img[80:180]) and np.array_equal(out[:, :50], img[:, :50])
