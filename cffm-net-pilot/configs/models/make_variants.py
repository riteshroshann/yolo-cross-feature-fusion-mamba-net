"""Write the ablation and baseline model YAMLs from cffm-net-n.yaml, each changing one thing."""
from pathlib import Path

import yaml

HERE = Path(__file__).parent
BASE = (HERE / "cffm-net-n.yaml").read_text(encoding="utf-8")

YOLO26_HEAD = """head:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 6], 1, Concat, [1]]
  - [-1, 2, C3k2, [512, True]]
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 4], 1, Concat, [1]]
  - [-1, 2, C3k2, [256, True]]
  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 13], 1, Concat, [1]]
  - [-1, 2, C3k2, [512, True]]
  - [-1, 1, Conv, [512, 3, 2]]
  - [[-1, 10], 1, Concat, [1]]
  - [-1, 1, C3k2, [1024, True, 0.5, True]]
  - [[16, 19, 22], 1, Detect, [nc]]
"""


def variant(name, fusion=None, head=None, modality_dropout=None, **cfg):
    text = BASE
    if fusion:
        text = text.replace("fusion: cffm\n", f"fusion: {fusion}\n")
    for k, v in cfg.items():
        old = next(line for line in text.splitlines() if line.strip().startswith(f"{k}:"))
        indent = old[: len(old) - len(old.lstrip())]
        val = str(v).lower() if isinstance(v, bool) else v
        text = text.replace(old, f"{indent}{k}: {val}")
    if modality_dropout:
        text = text.replace("fusion: ", f"modality_dropout: {modality_dropout}\nfusion: ", 1)
    if head:
        text = text[: text.index("head:")] + head
    out = HERE / name
    out.write_text(text, encoding="utf-8")
    yaml.safe_load(out.read_text())
    print("wrote", out.name)


if __name__ == "__main__":
    variant("cffm-net-n-nogate.yaml", gate=False)
    variant("cffm-net-n-gconv.yaml", mixer="gconv")
    variant("cffm-net-n-nop2.yaml", head=YOLO26_HEAD)
    variant("two-stream-concat-n.yaml", fusion="concat", head=YOLO26_HEAD)
    variant("two-stream-concat-p2-n.yaml", fusion="concat")
    variant("cffm-net-n-v2.yaml", head=YOLO26_HEAD, modality_dropout=0.15)
    variant("two-stream-concat-n-md.yaml", fusion="concat", head=YOLO26_HEAD, modality_dropout=0.15)
