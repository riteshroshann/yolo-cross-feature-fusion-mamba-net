"""Paired loading: one visible path in, a registered 4-channel pair out."""
import numpy as np
import pytest

from cffm import data as cdata


@pytest.fixture(scope="module")
def synth(tmp_path_factory):
    return cdata.make_synthetic(tmp_path_factory.mktemp("synth") / "synthetic", n_train=4, n_val=2, imgsz=96)


def test_thermal_path_mapping():
    assert cdata.thermal_path("/d/x/images/visible/train/0001.jpg") == "/d/x/images/infrared/train/0001.jpg"
    with pytest.raises(ValueError):
        cdata.thermal_path("/d/x/images/train/0001.jpg")


def test_read_pair_returns_four_channels(synth):
    p = next((synth / "images" / "visible" / "train").glob("*.jpg"))
    pair = cdata.read_pair(str(p))
    assert pair.shape == (96, 96, 4) and pair.dtype == np.uint8


def test_patch_is_off_outside_paired_datasets(synth):
    import cv2
    import ultralytics.data.base as base

    p = next((synth / "images" / "visible" / "train").glob("*.jpg"))
    assert base.imread(str(p), flags=cv2.IMREAD_COLOR).shape[2] == 3


def test_paired_dataset_yields_four_channel_tensors(synth):
    import yaml
    from ultralytics.cfg import get_cfg

    d = yaml.safe_load((synth / "data_paired.yaml").read_text())
    d["nc"] = len(d["names"])
    cfg = get_cfg(overrides={"imgsz": 96, "mosaic": 1.0})
    ds = cdata.build_dataset(cfg, str(synth / d["train"]), batch=2, data=d, mode="train")
    assert isinstance(ds, cdata.PairedYOLODataset)
    item = ds[0]
    assert item["img"].shape[0] == 4
    assert len(item["cls"]) > 0
