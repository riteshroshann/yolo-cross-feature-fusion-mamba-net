# CFFM-Net pilot

![CFFM-Net on a night crossing](results/showcase/hero_llvip.jpg)

Code and notebooks for the two-day pilot of **CFFM-Net** (Cross-Feature Fusion Mamba Network): a dual-stream
YOLO26 detector that fuses visible and thermal images with a selective state-space scan. In that scan, the
discretisation step of every token is scaled by how reliable its sensor is at its location. The pilot trains
the still-image model and its baselines and ablations on LLVIP and M3FD. It then decides five pre-registered
hypotheses. The novelty, the limitations of prior work it addresses and the literature survey are in
`docs/CFFM-Net_Pilot_Novelty.pdf` inside the zip (`../docs/novelty/` in the repository).

Everything is built to run on **Kaggle with two T4 GPUs**. It also runs on a local NVIDIA GPU.

## What is in the box

```
cffm-net-pilot/
  notebooks/
    phase0_environment/        00 setup and checks · 01 data acquisition · 02 data conversion
    phase1_still_images/       03 data audit · 04 single-sensor baselines · 05 two-stream baseline
                               06 inside CMFM · 07 train CFFM-Net · 08 ablations
                               08b head control · 08c, 08d round 2 (modality dropout)
                               09 degradation probe and latency · 15 results, hypotheses, figures
                               16 showcase: boards, camera failures, tracking, any photo
  src/cffm/                    the package the notebooks import
    scan.py                    selective scan: fused CUDA kernel or portable two-pass PyTorch scan
    blocks.py                  ReliabilityHead, OffsetAlign, GatedCrossScan, CMFM, baseline fusers
    model.py                   DualStreamDetectionModel: two YOLO26 backbones + fusion + YOLO26 neck/head
    data.py                    paired 4-channel datasets for Ultralytics; LLVIP and M3FD converters
    train.py                   DualTrainer, DualValidator (AI-TOD size bins), evaluate, probe, latency
    pipeline.py                Kaggle glue: find data and checkpoints, train from the run plan
    report.py                  results tables and the pre-registered hypothesis decisions
    viz.py                     predictions and reliability maps for any visible + thermal pair
    hud.py                     analysis boards: bracketed subjects, thermal and trust panels, trails
    showcase.py                scene picking, high-res pairs, ByteTrack sequences, inline display
    tsm.py                     Temporal State Memory (Phase 2; tested, not used in the pilot)
    env.py                     platform detection, paths, GPU report
  configs/pilot.yaml           every run and its settings (epochs, batch, data, model)
  configs/models/*.yaml        CFFM-Net, its ablations and the two-stream baseline
  tests/                       78 tests (65 run anywhere; the fused-kernel ones need a GPU with Triton)
  docs/figures/                every diagram as TikZ source: YOLO and Mamba references, CFFM-Net, training, scope
  tools/build_notebooks.py     regenerates all notebooks from one script
  tools/make_zip.py            packs the project into cffm-net-pilot.zip for Kaggle
  tools/kaggle_run.py          uploads the code and pushes and watches the notebooks via the Kaggle API
  tools/run_notebooks.py       executes notebooks locally and cleans their outputs for GitHub
  tools/build_demo.py          fills demo/ with sample pairs, photos and weights
  demo/app.py                  Gradio app: upload a pair or any photo, get the analysis board
  web/                         static front-end for Vercel that calls the demo's API
  results/                     pilot tables and figures; results/showcase holds the boards
  weights/yolo26n.pt           COCO-pretrained YOLO26-n; in the zip, not in git (notebook 00 downloads it)
```

## Demo

```bash
python tools/build_demo.py     # sample pairs, photos and weights into demo/
python demo/app.py             # http://127.0.0.1:7860
```

The web page in `web/` is a static site: serve it (`python -m http.server -d web 8000`) or deploy the folder on
Vercel, and point its backend field at the running app or a Hugging Face Space built with
`python tools/build_demo.py --space`.

## Running locally

```bash
python tools/run_notebooks.py          # every notebook, in order, outputs saved in place
python tools/run_notebooks.py 16       # just the showcase
```

## Running on Kaggle

**1. Upload the code.** Go to *Datasets → New Dataset* and upload `cffm-net-pilot.zip`. Name it
`cffm-net-pilot` and keep it private. Kaggle unzips it. To rebuild the zip after changing anything, run
`python tools/make_zip.py`. It writes `../dist/cffm-net-pilot.zip` and leaves out data, runs and caches.

**2. Upload the data.** Make two more private datasets from the official archives:

* LLVIP from <https://bupt-ai-cz.github.io/LLVIP/>;
* M3FD from <https://github.com/JinyuanLiu-CV/TarDAL> (the `M3FD_Detection` part).

The code finds them by their folder layout, so the names do not matter. Notebook 01 lists alternatives.

**3. Make the notebooks.** For each `.ipynb`, go to *Code → New Notebook → File → Import Notebook* and upload
the file. In the notebook settings, use:

* *Accelerator*: **GPU T4 x2**;
* *Internet*: **On** (needed to `pip install` Ultralytics);
* *Add Input*: `cffm-net-pilot`, plus LLVIP and M3FD for the notebooks that need data.

**4. Run in this order.**

| Notebook | Attach | Time on 2 x T4 | Produces |
|---|---|---|---|
| 00 environment setup | code | 10 to 15 min | go / no-go, measured speed and memory |
| 01 data acquisition | code, LLVIP, M3FD | minutes | verified raw data |
| 02 data conversion | code, LLVIP, M3FD | about 5 min | `data/llvip`, `data/m3fd` |
| 03 data audit | code, LLVIP, M3FD | about 5 min | sizes, darkness, alignment |
| 04 single-sensor baselines | code, LLVIP | 1.5 to 2 h | 2 trained runs |
| 05 two-stream baseline | code, LLVIP, M3FD | 2 to 2.5 h | 2 trained runs |
| 06 inside CMFM | code | 2 min | checks, no training |
| 07 train CFFM-Net | code, LLVIP, M3FD | 3 to 3.5 h | 2 trained runs |
| 08 ablations | code, LLVIP | 4 to 5 h | 2 (or 3) trained runs |
| 09 probe and latency | code, LLVIP, outputs of 04 05 07 08 | 1 to 1.5 h | probe and latency results |
| 15 results | code, outputs of 04 05 07 08 09 | minutes | tables, verdicts, figures |

Notebooks 04, 05, 07 and 08 are independent. If your Kaggle account allows more than one GPU session at a time,
they can run in parallel. Each fits well inside Kaggle's 12-hour session limit.

**5. Keep the outputs.** After a training notebook finishes, click *Save Version → Save & Run All (Commit)*.
Its `runs/` folder then becomes an output that notebooks 09 and 15 can attach (*Add Input → Your Work*).

**If a session dies mid-run,** run the notebook again. Finished runs are reused, and an unfinished run resumes
from its last checkpoint. In a fresh session, first attach the earlier version's output (*Add Input → Your
Work*); the checkpoints are picked up from there.

## Running locally (optional)

```bash
python -m venv .venv && .venv/Scripts/activate          # Linux/macOS: source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -e ".[dev]"
python -m pytest -q                                      # 65 pass, 13 skip without Triton or a GPU
CFFM_SMOKE=1 python -m pytest tests/test_smoke_train.py  # one-epoch end-to-end check on fake data
```

Put the raw datasets under `data/raw/` and open the notebooks from `notebooks/`. On Windows, the fused
`mamba_ssm` kernel is not available, so the PyTorch scan is used automatically.

## Changing the experiment

* **Epochs, batch sizes, which runs exist, and dataset options** live in `configs/pilot.yaml`. Every notebook
  reads it.
* **Model variants** live in `configs/models/`. To regenerate the ablation YAMLs after editing
  `cffm-net-n.yaml`, run `python configs/models/make_variants.py`. The gated-convolution control is matched to
  CFFM-Net at 18.30 against 18.39 GFLOPs, with the scan's element-wise cost included.
* **Notebook text and code** are generated by `tools/build_notebooks.py`. Edit that file and rerun it, so the
  eleven notebooks stay consistent.

## What the tests guarantee

* The two-pass PyTorch scan equals the textbook recurrence, in values and gradients, with and without
  checkpointing. It also equals the fused kernel when that is installed.
* A token with step 0 cannot write into the state. A sensor with zero reliability therefore cannot change the
  visible outputs of the cross scan.
* The scan interleaves visible and thermal tokens, and all four directions map back exactly.
* A fresh CMFM block is exactly a reliability-weighted average, and the offset alignment starts as the
  identity.
* Every model config builds and runs, and COCO weights fill both backbones.
* Pairs load as registered 4-channel images. The converters produce the documented layout, and the run plan is
  internally consistent.
* The hypothesis decisions follow the pre-registered rules.
* Runs and datasets attached from earlier notebook versions come back as writable copies, and the copy with the
  most finished epochs is the one resumed.
* The Phase 2 Temporal State Memory holds, resets and warps as specified.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `Add the 'cffm-net-pilot' dataset...` | Attach the uploaded zip as an input. |
| `raw llvip not found` | Attach LLVIP, or use the download cell in notebook 01. |
| `pip install` fails | Turn *Internet* on in the notebook settings. |
| Two-GPU training hangs or errors | Train on one GPU: `pipeline.train_from_spec(s, env, device="0")`. |
| CUDA out of memory | Halve that run's `batch` in `configs/pilot.yaml`. Notebook 00 section 6 checks this up front. |
| Session ended mid-run | Run the notebook again; it resumes from the last checkpoint. |
| Warning about `grid_sampler_2d_backward` not being deterministic | Harmless. The alignment layer's backward pass is not bit-exact on GPU. Seeds still fix data order and initialisation. |

## Scope of the pilot

One seed per configuration, the smallest YOLO26 scale, 30 epochs, and two registered datasets. The pilot can
support or undermine the hypotheses but cannot establish them. Differences below about 1.5 AP points are
reported as inconclusive. The full plan (three seeds, small scale, RGBT-Tiny, temporal memory) is in the
research dossier, `../docs/dossier/CFFM-Net_Research_Dossier.pdf`.
