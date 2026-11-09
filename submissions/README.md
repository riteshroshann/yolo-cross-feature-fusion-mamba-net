# Submissions

Everything needed to submit CFFM-Net, organised by venue. Each venue folder has a `paper/` folder:

```
<venue>/paper/
    paper.pdf              the file you upload
    supplementary.pdf      conferences, IEEE TIP and TCSVT: live runtime, multi-camera layer, per-seed tables
    cover_letter.pdf       journals only: upload this
    cover_letter.txt       journals only: paste this into the submission form
    highlights.txt         Information Fusion and Pattern Recognition only
    source/                the LaTeX that builds the PDFs
```

The paper is *CFFM-Net: Reliability-Gated Mamba Fusion of Visible and Thermal Images for Detection, Tracking and Multi-Camera Analytics* (`../papers/source`). Journals and arXiv carry the full paper. Conference versions are condensed to the page limit, are anonymous, say "code will be released" in place of the repository link, and move the live runtime, multi-camera layer, per-seed tables, size bands, ablation and latency tables to `supplementary.pdf`. Every number comes from `numbers.tex`; no claim was changed.

## Map

| Venue | Type | Deadline | Pages | Status |
|---|---|---|---|---|
| CVPR 2027 (`conferences/CVPR_2027`) | Conference | Registration 10 Nov 2026, paper 16 Nov 2026 AoE, supplement 23 Nov 2026 AoE | 8 + references (10 total); supplement 8 | Ready, but the mandatory LLM usage statement is not yet in the PDF |
| ICCV 2027 (`conferences/ICCV_2027`) | Conference | See call | 8 + references (10 total); supplement 8 | Ready; kit is ICCV 2025, header reads the 2027 year |
| WACV 2028 (`conferences/WACV_2028`) | Conference | See call | 8 + references (10 total); supplement 8 | Ready; kit is WACV 2026 |
| BMVC 2027 (`conferences/BMVC_2027`) | Conference | See call | about 11.6 + references (15 total; limit 14 excluding references); supplement 11 | Ready; paper number in `\bmvcreviewcopy` is a dash until OpenReview assigns it |
| IEEE TIP (`journals/IEEE_TIP`) | Journal | Rolling | 13 (limit 13 at first submission); supplement 3 | Ready; per-seed, ablation, size-band, latency, live-runtime and multi-camera tables are in `supplementary.pdf` |
| IEEE TCSVT (`journals/IEEE_TCSVT`) | Journal | Rolling | 14 (limit 14); supplement 3 | Ready; per-seed, ablation, multi-camera and live-runtime tables are in `supplementary.pdf` |
| Information Fusion (`journals/Information_Fusion`) | Journal | Rolling | 35 (review layout, double spaced) | Ready, with highlights |
| Pattern Recognition (`journals/Pattern_Recognition`) | Journal | Rolling | 35 (limit 20 to 35 double spaced) | Ready, with highlights |
| arXiv (`arxiv`) | Preprint | None | 31 | Ready; upload the `.tex`, `.bbl` and figures from `source/` |

Limits and kit versions are in `TEMPLATE_NOTES.md`; the BMVC 2026 call (14 pages excluding references) was re-checked on 9 November 2026.

## Before uploading

- **CVPR 2027:** the call requires an LLM usage statement after the references and rejects papers without one. Write it yourself and add it to `source/conf_body.tex`, then rebuild.
- **ICCV, WACV:** replace the kits when the 2027 and 2028 kits appear, and set the paper ID macros.
- **BMVC:** put the OpenReview number in `\bmvcreviewcopy{...}` in `source/main.tex` and `source/supplementary.tex`.
- **Information Fusion, Pattern Recognition:** upload `highlights.txt` as the separate highlights file (five bullets, each 85 characters or fewer).
- **Journals:** the affiliation is "Independent Researcher"; change it in `main.tex` and the cover letter if you prefer another.
- **Anonymous venues:** the PDF text and metadata were checked and contain no name, email, affiliation or repository link.
- **Overfull boxes and references:** every PDF compiles with no undefined reference or citation and no `??`.

## Build

```
tectonic -X compile main.tex            run in each source/ folder
tectonic -X compile supplementary.tex   conferences
tectonic -X compile cover_letter.tex    journals
```

Engineered by Ritesh Roshan.
