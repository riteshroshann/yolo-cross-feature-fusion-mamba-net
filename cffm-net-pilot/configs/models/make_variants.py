"""Write the ablation and baseline model YAMLs from cffm-net-n.yaml, each changing one thing.

Variants: nogate (C1), gconv (C4), nop2 (C2) and the two-stream concat baseline.
"""
from pathlib import Path

import yaml

HERE = Path(__file__).parent
BASE = (HERE / "cffm-net-n.yaml").read_text(encoding="utf-8")

# YOLO26 head without the stride-4 branch (Ultralytics yolo26.yaml, 8.4.171)
YOLO26_HEAD = """head:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 6], 1, Concat, [1]] # cat backbone P4
  - [-1, 2, C3k2, [512, True]] # 13
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 4], 1, Concat, [1]] # cat backbone P3
  - [-1, 2, C3k2, [256, True]] # 16 (P3/8-small)
  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 13], 1, Concat, [1]] # cat head P4
  - [-1, 2, C3k2, [512, True]] # 19 (P4/16-medium)
  - [-1, 1, Conv, [512, 3, 2]]
  - [[-1, 10], 1, Concat, [1]] # cat head P5
  - [-1, 1, C3k2, [1024, True, 0.5, True]] # 22 (P5/32-large)
  - [[16, 19, 22], 1, Detect, [nc]] # Detect(P3, P4, P5)
"""


def variant(name, header, fusion=None, head=None, **cfg):
    text = BASE
    if fusion:
        text = text.replace("fusion: cffm\n", f"fusion: {fusion}\n")
    for k, v in cfg.items():
        old = next(line for line in text.splitlines() if line.strip().startswith(f"{k}:"))
        indent = old[: len(old) - len(old.lstrip())]
        comment = old.split("#", 1)[1] if "#" in old else ""
        val = str(v).lower() if isinstance(v, bool) else v
        text = text.replace(old, f"{indent}{k}: {val}" + (f"  #{comment}" if comment else ""))
    if head:
        text = text[: text.index("head:")] + head
    body = text.split("\n", 1)[1] if text.startswith("#") else text
    while body.startswith("#"):
        body = body.split("\n", 1)[1]
    out = HERE / name
    out.write_text(header + body, encoding="utf-8")
    yaml.safe_load(out.read_text())  # must still parse
    print("wrote", out.name)


if __name__ == "__main__":
    variant("cffm-net-n-nogate.yaml", "# CFFM-Net, reliability gating OFF (r = 1 everywhere). Ablation for C1.\n", gate=False)
    variant("cffm-net-n-gconv.yaml", "# CFFM-Net with the selective scan replaced by a gated depthwise\n"
            "# convolution (MambaOut-style control). Ablation for C4.\n", mixer="gconv")
    variant("cffm-net-n-nop2.yaml", "# CFFM-Net without the stride-4 head (standard YOLO26 head). Ablation for C2.\n",
            head=YOLO26_HEAD)
    variant("two-stream-concat-n.yaml", "# Naive two-stream baseline: two YOLO26 backbones, features concatenated\n"
            "# and squeezed by a 1x1 conv at P3..P5, standard YOLO26 head.\n", fusion="concat", head=YOLO26_HEAD)
    variant("two-stream-concat-p2-n.yaml", "# Two-stream concat with CFFM-Net's stride-4 head: isolates the fusion "
            "from the head.\n", fusion="concat")
