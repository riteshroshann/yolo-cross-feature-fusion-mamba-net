"""Train runs from configs/pilot.yaml on this machine's GPU: python tools/train_local.py NAME [batch]."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cffm import env as cenv, pipeline


def main(name, batch=8):
    env = cenv.setup(verbose=False)
    plan = pipeline.load_plan(env.project)
    spec = pipeline.run_spec(plan, name)
    out = pipeline.train_from_spec(spec, env, device=0, batch=int(batch), name=f"{name}_b{batch}", workers=6)
    print(name, "done", out)


if __name__ == "__main__":
    main(*sys.argv[1:])
