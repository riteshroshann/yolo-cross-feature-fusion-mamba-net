"""Params, GFLOPs and FP16 batch-1 latency of the final LLVIP models on this machine's GPU, into results/latency_local.json."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch

from cffm.train import _load, gflops, latency_ms

RUNS = {
    "yolo26n_thermal": ROOT / "runs/llvip_yolo26n_thermal/weights/best.pt",
    "concat_md": ROOT / "runs/llvip_concat_md_e80/weights/best.pt",
    "concat_ft": ROOT.parent / "dist/out/08k-v3-llvip/runs/llvip_concat_ft_e80/weights/best.pt",
    "cffm_v2": ROOT / "runs/llvip_cffm_v2_e80/weights/best.pt",
    "cffm_v3_norel": ROOT.parent / "dist/out/08k-v3-llvip/runs/llvip_cffm_v3_norel_e80/weights/best.pt",
    "cffm_v3": ROOT.parent / "dist/out/08k-v3-llvip/runs/llvip_cffm_v3_e80/weights/best.pt",
}


def main():
    out = {"device": torch.cuda.get_device_name(0), "protocol": "FP16, batch 1, 640x512, 50 warm-up, 300 timed, 20 ms pause"}
    for name, path in RUNS.items():
        m = _load(path)
        ch = 3 if name.startswith("yolo26n") else 4
        row = {"params_M": sum(p.numel() for p in m.parameters()) / 1e6, "GFLOPs": gflops(m)}
        row |= latency_ms(m, imgsz=(512, 640), channels=ch, device="cuda:0", half=True, pause_s=0.02)
        out[name] = row
        print(f"{name:16s} {row['params_M']:.2f}M {row['GFLOPs']:.1f} GFLOPs {row['median_ms']:.2f} ms (p95 {row['p95_ms']:.2f})")
        del m
        torch.cuda.empty_cache()
    (ROOT / "results/latency_local.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
