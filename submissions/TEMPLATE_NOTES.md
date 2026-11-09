# Venue templates for CFFM-Net

Each venue folder holds a `template/` directory with the vendored class, style and bibliography files and a minimal `main.tex` (title, author block in the venue's form, abstract, one section, one `\rule` figure placeholder, one table, one bibliography entry). No paper content is in them. Every skeleton compiles with `tectonic -X compile main.tex` run inside its `template/` folder (checked 7 November 2026, Tectonic 0.17.0). Paths below are under `submissions/`.

Conference skeletons build the review (anonymous) version. Their author block is in the file but the style prints "Anonymous" in review mode, so switch to the camera-ready option only after acceptance.

| Venue | Folder | Class | Page limit | Anonymity | Deadline | Compile |
|---|---|---|---|---|---|---|
| CVPR 2027 | `conferences/CVPR_2027` | `cvpr.sty` on `article`, CVPR 2027 kit | 8 pages plus references and the mandatory LLM usage statement ([author guidelines](https://cvpr.thecvf.com/Conferences/2027/AuthorGuidelines)) | Double blind, `[review]` option | Registration 10 Nov 2026, submission 16 Nov 2026, supplement 23 Nov 2026, all AoE ([call](https://cvpr.thecvf.com/Conferences/2027/CallForPapers)) | Passes |
| ICCV 2027 | `conferences/ICCV_2027` | `iccv.sty`, ICCV 2025 kit (latest) | 8 pages plus references (kit text; 2027 call not published) | Double blind, `[review]` | Not announced | Passes |
| WACV 2028 | `conferences/WACV_2028` | `wacv.sty`, WACV 2026 kit (latest) | 8 pages excluding references (kit text; 2028 call not published) | Double blind, `[review]` | Not announced; WACV 2027 round 1 was 26 Jun 2026 and round 2 28 Aug 2026 ([WACV 2027 call](https://wacv.thecvf.com/Conferences/2027/CallForPapers)) | Passes |
| BMVC 2027 | `conferences/BMVC_2027` | `bmvc2k.cls`, BMVC 2026 kit (latest) | 14 pages excluding references ([BMVC 2026 call](https://bmvc2026.bmva.org/calls/call-for-papers/)) | Double blind, `\bmvcreviewcopy{??}` | Not announced; BMVC 2026 paper deadline was 29 May 2026 ([dates](https://bmvc2026.bmva.org/dates/)) | Passes |
| IEEE TIP | `journals/IEEE_TIP` | `IEEEtran` `[journal]` | 13 double-column pages at first submission, 16 when revised; overlength charge of $220 a page beyond published page 10 ([SPS information for authors](http://signalprocessingsociety.org/publications-resources/information-authors)) | Single blind | Rolling | Passes |
| IEEE TCSVT | `journals/IEEE_TCSVT` | `IEEEtran` `[journal]` | 14 pages for a Transactions paper, charges from page 12 ([guidelines](https://ieee-cas.org/publication/tcsvt/guidelines-authors)) | Single blind | Rolling | Passes |
| Information Fusion | `journals/Information_Fusion` | `elsarticle` `[review,3p,times]` | No research-article page limit found; guide caps research articles at 50 references and surveys at 50 pages ([guide](https://www.sciencedirect.com/journal/information-fusion/publish/guide-for-authors)) | Check the guide; skeleton is not anonymised | Rolling | Passes |
| Pattern Recognition | `journals/Pattern_Recognition` | `elsarticle` `[review,3p,times]` | 20 to 35 single-column double-spaced pages ([guide](https://www.sciencedirect.com/journal/pattern-recognition/publish/guide-for-authors)) | Check the guide; skeleton is not anonymised | Rolling | Passes |
| arXiv | `arxiv` | `article`, `natbib`, `hyperref` | None | None | None | Passes |

The Information Fusion and Pattern Recognition limits come from search-result excerpts of the ScienceDirect guides, because ScienceDirect returned HTTP 403 to a direct fetch. Read both guides before submitting. The Pattern Recognition guide also states that appendices are outside the limit while counting them in the same sentence; ask the editorial office if it matters.

## Notes per venue

**CVPR 2027.** Kit is release `CVPR2027-v1` of `cvpr-org/author-kit` (28 Oct 2026). Build uses `main.bib` and `ieeenat_fullname.bst`. The 2027 call adds a mandatory LLM usage statement after the references; papers without it are rejected without review. The kit has no paper ID any more. The skeleton drops the kit's teaser figure and section files and keeps `preamble.tex`, which loads `cuted`, `currfile` and `caption`. Tectonic prints a repeated "Invalid UTF-8 byte at line 296" warning from `lineno.sty`; the PDF is correct and the warning can be ignored.

**ICCV 2027.** ICCV is held in odd years, but no 2027 kit or call is online (the 2027 pages return "Page not found"). The folder uses the ICCV 2025 kit from the official site ([zip](https://media.eventhosts.cc/Conferences/ICCV2025/ICCV2025-Author-Kit-Feb.zip)). The style needs `\def\paperID{*****}` and `\confYear` set in `main.tex`. The skeleton shows "ICCV 2025" in the header until the 2027 kit replaces `iccv.sty`. Same `lineno.sty` warning as CVPR.

**WACV 2028.** The WACV 2027 call says "Templates to be provided" and the 2027 kit URL returns 404, so the folder uses the WACV 2026 kit ([zip](https://media.eventhosts.cc/Conferences/WACV2026/wacv-2026-author-kit-template.zip)). The paper ID macro is `\wacvPaperID`, and the style option must name a track (`algorithms` or `applications`); the skeleton uses `algorithms`. The header reads "WACV 2026" until the kit is replaced. Same `lineno.sty` warning.

**BMVC 2027.** Files come from the BMVA repository `BritishMachineVisionAssociation/BMVCTemplate` (update of 2 Apr 2026, the BMVC 2026 kit). The class is `bmvc2k` with `natbib` bundled as `bmvc2k_natbib.sty`; bibliography file is `egbib.bib`. Replace `??` in `\bmvcreviewcopy` with the OpenReview paper number. The kit says pdfLaTeX gives the best results; Tectonic (XeTeX) compiled the skeleton without errors. The affiliation line "Independent Researcher" is a stand-in for the real affiliation.

**IEEE TIP and TCSVT.** Both use `IEEEtran.cls` and `IEEEtran.bst` version from CTAN, `[journal]` option. The two skeletons differ only in the `\markboth` running head and the footnote text. The affiliation footnote reads "independent researcher"; replace it with the real affiliation. The CAS and SPS societies do not ship a separate class, so IEEEtran is the template. Tectonic prints harmless underfull-box warnings from the title block.

**Information Fusion and Pattern Recognition.** CTAN ships `elsarticle` only as `elsarticle.dtx` and `.ins`, and Tectonic cannot run the `.ins` (docstrip output is discarded and xdvipdfmx fails). `elsarticle.cls` was therefore extracted by hand from lines 90 to 1527 of `elsarticle.dtx` (the `class` block), dropping comment lines, with the licence header added back at the top. The result compiles and matches the version Tectonic's own bundle provides (3.5, 9 Jan 2026). Both folders also hold `elsarticle-num.bst` and `elsarticle-harv.bst`; the skeleton uses `elsarticle-num`. The `lineno` package is loaded for the review layout and gives the same UTF-8 warning as the CVF kits. Elsevier systems also accept a Word submission; LaTeX needs the source files uploaded as well as the PDF.

**arXiv.** Plain `article` class with `natbib` and `hyperref`, bibliography via `plainnat`. arXiv builds with pdfLaTeX, so upload the `.tex` plus the `.bbl` produced locally (arXiv does not run BibTeX), and use PDF, PNG or JPG figures. Do not put `\pdfoutput=1` in a file you compile with Tectonic: under XeTeX it makes `hyperref` select the pdfTeX driver and fail with "Undefined control sequence" in `hpdftex.def`. Add it only to the copy you upload, as the first line, if arXiv's autodetection ever needs it.

## Licences

None of the CVF or BMVA kits ship a licence file. `elsarticle` and `IEEEtran` carry the LaTeX Project Public License in their file headers, and the unmodified `README` files from each kit are kept next to the class files. The only edit to vendored code is the extraction and added header on `elsarticle.cls`.

## Open items

- Replace the ICCV 2025 and WACV 2026 kits when the 2027 and 2028 kits appear, then update the year and paper ID macros.
- Add the real affiliation in the BMVC and IEEE skeletons.
- Confirm the Information Fusion and Pattern Recognition limits against the live guides.

Engineered by Ritesh Roshan.
