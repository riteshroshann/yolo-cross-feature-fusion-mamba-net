"""CFFM-Net demo: upload a visible + thermal pair, or any photo, and see the Machine's analysis.

    python demo/app.py            # http://127.0.0.1:7860, GPU if available
"""
import sys
from pathlib import Path

import cv2
import gradio as gr
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src" if (HERE.parent / "src" / "cffm").is_dir() else HERE))

from ultralytics import YOLO

from cffm import hud, showcase as sc
from cffm.train import _load

DEV = "cuda:0" if torch.cuda.is_available() else "cpu"


def weights(name):
    for p in (HERE / "weights" / f"{name}.pt", HERE.parent / "runs" / name / "weights" / "best.pt",
              HERE.parent / "weights" / f"{name}.pt"):
        if p.exists():
            return str(p)
    raise FileNotFoundError(f"weights for {name} not found under demo/weights or runs/")


MODELS = {"Night pedestrians · LLVIP": _load(weights("llvip_cffm")).to(DEV),
          "City traffic · M3FD": _load(weights("m3fd_cffm")).to(DEV)}
COCO = YOLO(weights("yolo26n"))
SAMPLES = HERE / "samples"


def _log(a):
    df = sc.subject_log(a.det, a.names, a.share)
    return df.reset_index() if len(df) else None


def run_pair(visible, thermal, which, conf):
    if visible is None or thermal is None:
        raise gr.Error("Upload both images: a visible (colour) one and a thermal one of the same scene.")
    vis = cv2.cvtColor(visible, cv2.COLOR_RGB2BGR)
    ir = cv2.cvtColor(thermal, cv2.COLOR_RGB2GRAY) if thermal.ndim == 3 else thermal
    a = hud.analyze(MODELS[which], vis, ir, conf=conf)
    return a.board, _log(a), f"{len(a.det)} subjects · {a.ms:.0f} ms on {DEV.upper()}"


def run_photo(photo, conf):
    if photo is None:
        raise gr.Error("Upload a photo.")
    a = hud.analyze(COCO, cv2.cvtColor(photo, cv2.COLOR_RGB2BGR), conf=conf, title="YOLO26-N",
                    subtitle="COCO, 80 CLASSES (NOT CFFM-NET)")
    return a.board, _log(a), f"{len(a.det)} subjects · {a.ms:.0f} ms on {DEV.upper()}"


def pair_examples():
    rows = []
    for v in sorted((SAMPLES / "pairs").glob("*_visible.*")):
        t = v.with_name(v.name.replace("_visible", "_thermal"))
        if t.exists():
            rows.append([str(v), str(t), "City traffic · M3FD" if v.name.startswith("m3fd") else
                         "Night pedestrians · LLVIP", 0.3])
    return rows


CSS = """
.gradio-container {max-width: 1500px !important}
#title {font-family: 'JetBrains Mono', ui-monospace, monospace; letter-spacing: .04em}
#title h1 {font-size: 1.6rem; margin-bottom: 0}
#title p {color: #8a94a3; margin-top: .3rem}
"""
THEME = gr.themes.Base(primary_hue="yellow", neutral_hue="slate",
                       font=[gr.themes.GoogleFont("Inter"), "sans-serif"],
                       font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "monospace"]).set(
    body_background_fill="#08090c", body_background_fill_dark="#08090c", block_background_fill_dark="#0f1216",
    button_primary_background_fill="#ffcc00", button_primary_text_color="#08090c")

with gr.Blocks(title="CFFM-Net · The Machine's view") as app:
    gr.Markdown("# CFFM-NET // THE MACHINE'S VIEW\nCross-feature fusion of a visible and a thermal camera with a "
                "reliability-gated selective scan. Every subject is bracketed and numbered; the trust map shows "
                "where the model weighted each camera.", elem_id="title")
    with gr.Tabs():
        with gr.Tab("Visible + thermal (CFFM-Net)"):
            with gr.Row():
                vis = gr.Image(label="visible camera", type="numpy", height=260)
                ir = gr.Image(label="thermal camera", type="numpy", height=260)
                with gr.Column(scale=1, min_width=260):
                    which = gr.Radio(list(MODELS), value=list(MODELS)[0], label="model")
                    conf = gr.Slider(0.05, 0.9, value=0.3, step=0.05, label="confidence threshold")
                    go = gr.Button("Analyze", variant="primary")
            board = gr.Image(label="analysis", type="numpy", buttons=["download", "fullscreen"])
            status = gr.Markdown()
            log = gr.Dataframe(label="subjects", interactive=False)
            go.click(run_pair, [vis, ir, which, conf], [board, log, status], api_name="analyze_pair")
            ex = pair_examples()
            if ex:
                gr.Examples(ex, [vis, ir, which, conf], [board, log, status], fn=run_pair, cache_examples=False,
                            run_on_click=True, label="samples from the LLVIP and M3FD test sets")
        with gr.Tab("Any photo (YOLO26-n, COCO)"):
            with gr.Row():
                photo = gr.Image(label="photo", type="numpy", height=320)
                with gr.Column(scale=1, min_width=260):
                    conf2 = gr.Slider(0.05, 0.9, value=0.3, step=0.05, label="confidence threshold")
                    go2 = gr.Button("Analyze", variant="primary")
                    gr.Markdown("CFFM-Net needs a thermal camera, so ordinary photos go to the general 80-class "
                                "YOLO26-n, drawn the same way.")
            board2 = gr.Image(label="analysis", type="numpy", buttons=["download", "fullscreen"])
            status2 = gr.Markdown()
            log2 = gr.Dataframe(label="subjects", interactive=False)
            go2.click(run_photo, [photo, conf2], [board2, log2, status2], api_name="analyze_photo")
            photos = [[str(p), 0.3] for p in sorted((SAMPLES / "photos").glob("*.jpg"))]
            if photos:
                gr.Examples(photos, [photo, conf2], [board2, log2, status2], fn=run_photo, cache_examples=False,
                            run_on_click=True)

if __name__ == "__main__":
    app.queue().launch(theme=THEME, css=CSS, share="--share" in sys.argv,
                       server_name="0.0.0.0" if "--public" in sys.argv else "127.0.0.1")
