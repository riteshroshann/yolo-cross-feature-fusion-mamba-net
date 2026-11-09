# CFFM-Net numbers

`numbers.tex` holds every number the paper cites as a `\newcommand`, and `tables/*.tex` are built only from those macros and from `tools/sota.json` (published figures copied from the cited papers). Regenerate both with the repo venv:

    .venv\Scripts\python.exe papers\source\tools\collect_numbers.py

Output is deterministic. The script prints a cross-check report after the macro count. Nothing is appended by hand: significance tests, margins, summaries and latency macros are all computed in the script. Tables need `booktabs` and are written for IEEEtran. A table cell with no macro prints a dash. `tools/make_fault_figure.py` redraws `figures/fig_fault_margin.pdf` from `numbers.tex`. Hand-set protocol parameters live in `parts/config.tex`.

## Macro scheme

Names are alphabetic camelCase: prefix, dataset, model, seed or statistic, metric. Datasets are `Llvip` and `Mfd`. Numbers in names are spelled out (`Vtwo`, `SeedOne`, `Three`). mAP-style values are percent with one decimal; margins are percentage points with a math sign.

| prefix | content |
|---|---|
| `res` | one run: `resLlvipCffmVtwoSeedOneMap`; seed statistics use `Mean`/`Std` (sample std, n-1) in place of the seed; `SeedCount` gives n |
| `mar` | margin A minus B, per seed, `Mean` and `Std` over seeds |
| `sig` | v2 over concat, paired t-test with df 2 (t = 4.303 for the 95 percent interval), `P`, `CiLow`, `CiHigh`, sign-test `Wins`, `SignP`, `SignTwoSidedP`; `Small` variants for small-object AP |
| `gain` | M3FD fusion over the matched single-camera baselines (one seed each): `gainMfd{Model}Over{Best,Vis,Thm}{Mean,Std,Min}Map` |
| `flt` | fault benchmark. One seed gives `...Map`; several give `...MeanMap` and `...StdMap`. `AvgSevThree` summaries are means over the eleven faults at severity three; `Margin` is v2 minus concat; `VtwoVsConcatFt` compares with the fault-trained concat baseline at seed zero |
| `lat` | round-one parameters, GFLOPs, ms (`latLlvip...`, earlier hardware); `latLocal...` is the final re-timing on the idle local GPU |
| `prb` | degradation probe, LLVIP, round-one models |
| `hyp` | pilot hypothesis statement, evidence and verdict as text |
| `cam` | phase four: detection, fusion parameters, tracking, counting, crossings, dwell |
| `kst` | KAIST single-frame miss rate (`Mr`) and AP50 (`ApFifty`), concat and CFFM-Net v2 |
| `mem` | Temporal State Memory: miss rate by condition, image or memory mode, subset; `memChange...` is memory minus image |
| `trk` | tracking on four KAIST sequences: MOTA, IDF1 means, switches, wins; `trkFaultCost...` |
| `liv` | live runtime: soak at 30 and 20 fps, link-fault run with monitor on and off, monitor timing |
| `sota` | ranges of the published numbers quoted in prose, from `tools/sota.json` |

Metric tokens: `Map` mAP50-95, `MapFifty` mAP50, `Precision`, `Recall`, `ApSmall`/`ApMedium`/`ApLarge` (COCO size bands), `ApBandVeryTiny`/`ApBandTiny`/`ApBandSmall` (M3FD sub-bands). Model tokens: `YoloVis`, `YoloThm`, `ConcatPlain`, `ConcatPtwo`, `CffmVone` (+`Nogate`, `Gconv`, `Noptwo`), `ConcatMd`, `CffmVtwo`, `ConcatFt`, `CffmVthree`, `CffmVthreeNorel`.

## Tables

| file | shows | sources |
|---|---|---|
| `tab_main` | mAP and mAP50, v2 against concat (3-seed mean and std), margins, single-camera rows (LLVIP 30 epochs, M3FD 80 epochs matched), margin over the best single camera on M3FD | `runs/faults/*_e80/clean`, `dist/out/08g-08j`, `runs/eval/{llvip,m3fd}_yolo26n_*` |
| `tab_scale` | small, medium, large AP | same |
| `tab_ablation`, `tab_rounds` | round one to four variants and margins | `runs/eval/*`, `dist/out/08k`, `08l` |
| `tab_fault` | severity-three AP for 11 faults: v2, concat, fault-trained concat; LLVIP seed 0, M3FD mean and std over three seeds (concat FT seed 0) | `results/faults.csv` |
| `tab_probe` | LLVIP degradation probe | `results/probe.csv` |
| `tab_latency` | params, GFLOPs, median and p95 latency, idle RTX 4050 Laptop GPU | `results/latency_local.json` |
| `tab_multicam` | phase four detection, tracking, counting | `results/phase4/phase4_summary.json` |
| `tab_kaist` | KAIST single-frame miss rate and AP50 | `results/phase2/stills.json` |
| `tab_memory` | memory against image mode under four failure conditions, and tracking | `results/phase2/memory.json`, `tracking.json` |
| `tab_live`, `tab_monitor` | soak, link-fault run, monitor timing | `results/phase3/*.json` |
| `tab_sota`, `tab_sota_kaist` | published results beside ours, protocol differences stated | `tools/sota.json`, `sota_numbers.md` |

## Runs

Found: seeds 0, 1, 2 for v2 and concat-md at 80 epochs on both datasets; v3, v3 without reliability and fault-trained concat at 80 epochs on both datasets (seed 0); all round-one runs on LLVIP; round-one concat and CFFM on M3FD; matched 80-epoch single-camera YOLO26-n on M3FD (`runs/eval/m3fd_yolo26n_{visible,thermal}_b32`); fault benchmark for v2 and concat-md (M3FD three seeds, LLVIP seed 0) and fault-trained concat (seed 0, both datasets); KAIST phases two and three.

Missing:
- Round-one ablations (no gate, gated conv, no P2, concat+P2) on M3FD.
- Round two (30-epoch v2 and concat-md): no metrics file is stored locally.
- Fault benchmark for v3 and its controls; v3 seeds 1 and 2.
- LLVIP fault benchmark at seeds 1 and 2.
- A stateless adapter control for the memory, and the scan against a gated convolution in the v2 recipe over seeds.
- Device model of the phase-three runs (not recorded in the result files).

## Inconsistencies

- `results/*.csv` and `results/pilot_v1/*.csv` disagree (main_results, probe, hypotheses) by up to 0.0011 AP. The top-level files match `runs/eval/*/metrics.json` to 4 decimals, so they are used; `pilot_v1` is ignored.
- `faults.csv` clean rows match the clean `metrics.json` of each run to within 3e-4 (seed-two M3FD v2 differs by 2e-4); the tolerance of the check is 3e-4.
- `visible_drop` and `thermal_drop` give identical AP at severities one to three, so only severity three is tabulated.
- Margins are means of unrounded per-seed differences, so a printed margin can differ by 0.1 from the difference of the printed means.
- Round-one latency (`latLlvip...`) comes from an earlier session on unrecorded hardware and is not comparable with `latLocal...`. The earlier provisional v2 and concat-md timings (ratio 1.6) are superseded by the final re-timing (ratio 1.36).
- Pilot hypothesis texts (`hyp`) come from the round-one 30-epoch runs and are not updated for the 80-epoch results.
- The KAIST miss rate is our implementation of the Caltech protocol on the original annotation files, so it is not comparable with published figures.
