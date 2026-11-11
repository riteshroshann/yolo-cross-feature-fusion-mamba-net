"""Export the demo models to ONNX in web/models/ so the site can run them in the browser.

    python tools/export_onnx.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cffm.train import _load

OUT = ROOT / "web" / "models"
MODELS = {"llvip_cffm": 4, "m3fd_cffm": 4, "yolo26n": 3}


def export(name: str, channels: int) -> dict:
    import onnxruntime as ort

    model = _load(ROOT / "demo" / "weights" / f"{name}.pt").float()
    x = torch.rand(1, channels, 640, 640)
    with torch.no_grad():
        ref = model(x)
    ref = (ref[0] if isinstance(ref, (tuple, list)) else ref).numpy()
    path = OUT / f"{name}.onnx"
    torch.onnx.export(model, x, str(path), input_names=["images"], output_names=["detections"], opset_version=17,
                      dynamo=False)
    got = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"]).run(None, {"images": x.numpy()})[0]
    keep = ref[0, :, 4] > 0.25
    err = float(np.abs(got[0][keep] - ref[0][keep]).max()) if keep.any() else 0.0
    print(f"{name}: {path.stat().st_size / 1e6:.1f} MB, output {got.shape}, max error on kept boxes {err:.4f}")
    return {"channels": channels, "names": [model.names[i] for i in range(len(model.names))],
            "end2end": bool(getattr(model, "end2end", False)), "mb": round(path.stat().st_size / 1e6)}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    meta = {name: export(name, channels) for name, channels in MODELS.items()}
    (OUT / "meta.json").write_text(json.dumps(meta), encoding="utf-8")


if __name__ == "__main__":
    main()
