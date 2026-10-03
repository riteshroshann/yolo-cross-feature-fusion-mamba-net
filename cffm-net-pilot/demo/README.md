---
title: CFFM-Net
colorFrom: yellow
colorTo: gray
sdk: gradio
app_file: app.py
pinned: false
---

# CFFM-Net demo

Upload a visible + thermal pair (CFFM-Net) or any photo (YOLO26-n, COCO) and get the analysis board.

```bash
python tools/build_demo.py      # samples and weights into demo/
python demo/app.py              # http://127.0.0.1:7860
```

To deploy as a Hugging Face Space: `python tools/build_demo.py --space`, then push `dist/hf-space/` to a new
Gradio Space. The front-end in `web/` (Vercel) calls the Space's `analyze_pair` and `analyze_photo` endpoints.

Sample pairs come from the LLVIP and M3FD test sets (research use, see their licences).
