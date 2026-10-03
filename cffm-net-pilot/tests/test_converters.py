"""Converters and pipeline helpers, on tiny fake datasets in the official folder layouts."""
import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from cffm import data as cdata
from cffm import pipeline
from cffm.env import Env


def _voc(path, objects, w, h):
    objs = "".join(f"<object><name>{n}</name><bndbox><xmin>{a}</xmin><ymin>{b}</ymin><xmax>{c}</xmax>"
                   f"<ymax>{d}</ymax></bndbox></object>" for n, a, b, c, d in objects)
    path.write_text(f"<annotation><size><width>{w}</width><height>{h}</height></size>{objs}</annotation>")


@pytest.fixture()
def fake_llvip(tmp_path):
    root = tmp_path / "raw" / "LLVIP"
    (root / "Annotations").mkdir(parents=True)
    rng = np.random.default_rng(0)
    for split, n in (("train", 6), ("test", 3)):
        for k in range(n):
            stem = f"{split[:2]}{k:04d}"
            for mod in ("visible", "infrared"):
                d = root / mod / split
                d.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(d / f"{stem}.jpg"), rng.integers(0, 255, (1024, 1280, 3), dtype=np.uint8))
            _voc(root / "Annotations" / f"{stem}.xml", [("person", 100, 200, 160, 360)], 1280, 1024)
    return tmp_path


@pytest.fixture()
def fake_m3fd(tmp_path):
    root = tmp_path / "raw" / "M3FD_Detection"
    for d in ("Vis", "Ir", "Annotation"):
        (root / d).mkdir(parents=True)
    rng = np.random.default_rng(1)
    for k in range(10):
        for d in ("Vis", "Ir"):
            cv2.imwrite(str(root / d / f"{k:05d}.png"), rng.integers(0, 255, (768, 1024, 3), dtype=np.uint8))
        _voc(root / "Annotation" / f"{k:05d}.xml", [("People", 10, 10, 30, 60), ("Car", 300, 300, 420, 380),
                                                    ("Unknown", 0, 0, 5, 5)], 1024, 768)
    return tmp_path


def make_env(tmp):
    e = Env("local", Path(tmp), Path(tmp) / "data", Path(tmp) / "raw", Path(tmp) / "runs", Path(tmp) / "w", "cpu")
    for p in (e.data, e.runs, e.weights):
        p.mkdir(parents=True, exist_ok=True)
    return e


def test_llvip_found_converted_and_loadable(fake_llvip):
    env = make_env(fake_llvip)
    assert pipeline.locate_raw("llvip", env) == fake_llvip / "raw" / "LLVIP"
    out = pipeline.prepare("llvip", env, {"datasets": {"llvip": {"train_stride": 2, "val_stride": 1}}})
    card = json.loads((out / "dataset_card.json").read_text())
    assert card["images"] == {"train": 3, "val": 3}
    im = cv2.imread(str(next((out / "images" / "visible" / "train").glob("*.jpg"))))
    assert im.shape[:2] == (512, 640)
    label = next((out / "labels" / "visible" / "train").glob("*.txt")).read_text().split()
    assert label[0] == "0" and abs(float(label[3]) - 60 / 1280) < 1e-4
    d = yaml.safe_load((out / "data_paired.yaml").read_text())
    assert d["paired"] and d["channels"] == 4 and d["test"] == "images/visible/val"
    assert (out / "val_mini_visible.txt").read_text().startswith("./images/visible/val/")
    mtime = (out / "dataset_card.json").stat().st_mtime
    pipeline.prepare("llvip", env)
    assert (out / "dataset_card.json").stat().st_mtime == mtime


def test_m3fd_split_classes_and_unknowns(fake_m3fd):
    env = make_env(fake_m3fd)
    out = pipeline.prepare("m3fd", env, {"datasets": {"m3fd": {"val_frac": 0.2, "seed": 0}}})
    card = json.loads((out / "dataset_card.json").read_text())
    assert card["images"] == {"train": 8, "val": 2} and card["names"] == cdata.M3FD_NAMES
    labels = [l.split()[0] for f in (out / "labels" / "visible").rglob("*.txt") for l in f.read_text().splitlines()]
    assert set(labels) == {"0", "1"}


def test_locate_raw_tells_llvip_and_m3fd_apart(fake_llvip, fake_m3fd):
    env = make_env(fake_llvip)
    assert pipeline.locate_raw("llvip", env) == fake_llvip / "raw" / "LLVIP"
    assert pipeline.locate_raw("m3fd", env) == fake_m3fd / "raw" / "M3FD_Detection"


def test_run_plan_is_consistent():
    root = Path(__file__).resolve().parents[1]
    plan = pipeline.load_plan(root)
    names = [r["name"] for r in plan["runs"]]
    assert len(names) == len(set(names))
    for r in plan["runs"]:
        s = pipeline.run_spec(plan, r["name"])
        assert s["batch"] % 2 == 0, "batches must split evenly over 2 GPUs"
        if s["model"].startswith("configs/"):
            assert (root / s["model"]).exists(), s["model"]
    for run in plan["probe"]["runs"]:
        assert run in names


def test_find_checkpoints_and_metrics(tmp_path):
    env = make_env(tmp_path)
    (env.runs / "a" / "weights").mkdir(parents=True)
    (env.runs / "a" / "weights" / "best.pt").write_bytes(b"x")
    (env.runs / "eval" / "a").mkdir(parents=True)
    (env.runs / "eval" / "a" / "metrics.json").write_text(json.dumps({"mAP50-95(B)": 0.5}))
    assert pipeline.find_checkpoints(env)["a"].name == "best.pt"
    assert pipeline.find_metrics(env, "eval")["a"]["mAP50-95(B)"] == 0.5
