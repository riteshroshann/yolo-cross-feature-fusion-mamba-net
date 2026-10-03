# Licences

Copyright © 2026 Ritesh Roshan.

**Code: GNU AGPL v3.0 only** ([LICENSE](LICENSE)). This covers everything that runs: `cffm-net-pilot/src`, `tests`,
`tools`, `configs`, `demo` and `web`, the notebooks, and the build scripts under `docs/` and `study-prep/`. CFFM-Net
imports and extends Ultralytics YOLO, which is AGPL-3.0, and its model configurations are derived from the
Ultralytics YOLO26 configuration. AGPL-3.0 is therefore the licence the combined work requires.

**Documents and figures: CC BY 4.0** ([LICENSES/CC-BY-4.0.txt](LICENSES/CC-BY-4.0.txt)). This covers:
- the text of the READMEs;
- the PDFs and LaTeX sources in `docs/` and `study-prep/guide`;
- the diagrams in `cffm-net-pilot/docs/figures`;
- the result tables and plots in `cffm-net-pilot/results`.

Reuse requires attribution: *Ritesh Roshan, CFFM-Net,
https://github.com/riteshroshann/yolo-cross-feature-fusion-mamba-net*, or the citation in
[CITATION.cff](CITATION.cff).

## Third-party material

The following material stays under its owners' terms. Neither licence above covers it.

| material | where it appears | terms |
|---|---|---|
| LLVIP images, and boards or figures drawn from them | `results/showcase`, `web/assets`, `demo/samples`, `docs/figures/img` | [LLVIP terms of use](https://github.com/bupt-ai-cz/LLVIP/blob/main/Term%20of%20Use%20and%20License.md): non-commercial use only; cite Jia et al., ICCVW 2021 |
| M3FD images, and boards drawn from them | `results/showcase`, `web/assets`, `demo/samples` | terms of the [TarDAL](https://github.com/JinyuanLiu-CV/TarDAL) authors; cite Liu et al., CVPR 2022 |
| MOT17 frames | `tracking_mot17.gif`, `web/assets` | CC BY-NC-SA 3.0, [MOTChallenge](https://motchallenge.net); cite Milan et al., 2016 |
| `bus.jpg` and `zidane.jpg`, and the boards drawn from them | `photo_*`, `demo/samples`, `web/assets/samples` | Ultralytics, AGPL-3.0 |
| YOLO26 pretrained weights | not in this repository | Ultralytics, AGPL-3.0 |
| cited papers | not in this repository | copyright of their authors and publishers |

The papers are fetched from their original hosts by `papers/download_papers.py` and `study-prep/papers/fetch.py`.

Checkpoints trained here inherit the AGPL-3.0 of the YOLO26 weights they start from. Checkpoints trained on LLVIP are
also bound by its non-commercial terms. Commercial use of the Ultralytics components needs either full AGPL-3.0
compliance or an [Ultralytics Enterprise License](https://www.ultralytics.com/license).
