"""Kaggle glue: runs and datasets attached as read-only inputs must come back as writable copies."""
import os
import stat

from cffm import pipeline


def _fake_run(root, name, epochs):
    run = root / name
    (run / "weights").mkdir(parents=True)
    (run / "weights" / "last.pt").write_bytes(b"ckpt")
    (run / "results.csv").write_text("epoch\n" + "".join(f"{i}\n" for i in range(epochs)))
    for p in [*run.rglob("*")]:
        if p.is_file():
            os.chmod(p, stat.S_IREAD)                    # inputs on Kaggle are read-only
    return run


def test_prior_run_prefers_most_epochs(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "INPUT_ROOTS", [tmp_path])
    _fake_run(tmp_path / "v1" / "runs", "llvip_cffm", 12)
    longest = _fake_run(tmp_path / "v2" / "runs", "llvip_cffm", 20)
    _fake_run(tmp_path / "v3" / "runs", "llvip_cffm_nogate", 30)  # another run, must not match
    assert pipeline.find_prior_run("llvip_cffm") == longest
    assert pipeline.find_prior_run("m3fd_cffm") is None


def test_copy_writable(tmp_path):
    src = _fake_run(tmp_path / "input", "llvip_cffm", 3)
    dst = pipeline.copy_writable(src, tmp_path / "working" / "llvip_cffm")
    assert (dst / "weights" / "last.pt").read_bytes() == b"ckpt"
    assert all(os.access(p, os.W_OK) for p in [dst, *dst.rglob("*")])
