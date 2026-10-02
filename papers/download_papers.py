"""Download the open-access papers cited in the research dossier.

Every entry is keyed by its citation key in report/references.tex. arXiv
papers are fetched through the arXiv API, and the title returned by the API is
checked against a keyword before the PDF is saved, so that a wrong identifier
is caught instead of silently downloading the wrong paper. Entries without an
identifier are found by an arXiv title search. A few papers come from other
open-access hosts (CVF, MDPI, Nature, EUR-Lex).

Usage:  python papers/download_papers.py          (skips files already present)
Output: papers/<category>/<year>_<name>.pdf, papers/index.csv, papers/README.md
"""
import csv
import os
import re
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

ROOT = os.path.dirname(os.path.abspath(__file__))
UA = "cffm-net-paper-fetch/1.0 (research dossier; polite single-thread client)"
API = "http://export.arxiv.org/api/query?"
DELAY = 3.2  # arXiv asks for at least 3 seconds between requests
ATOM = {"a": "http://www.w3.org/2005/Atom"}

CATS = {
    "00": "00_source_papers",
    "01": "01_detectors",
    "02": "02_state_space_models",
    "03": "03_mamba_detectors",
    "04": "04_small_objects",
    "05": "05_rgbt_fusion",
    "06": "06_video_tracking_streaming",
    "07": "07_datasets",
    "08": "08_foundation_models",
    "09": "09_other_methods",
    "10": "10_law_and_ethics",
}


def E(key, cat, name, check=None, arxiv=None, url=None, search=None, year=None, note=None, title=None):
    return dict(key=key, cat=cat, name=name, check=check, arxiv=arxiv, url=url,
                search=search, year=year, note=note, title=title)


PAPERS = [
    # ---- detectors -------------------------------------------------------
    E("yolo26_2026", "01", "YOLO26", "YOLO26", arxiv="2606.03748"),
    E("wang2024yolov10", "01", "YOLOv10", "YOLOv10", arxiv="2405.14458"),
    E("tian2025yolov12", "01", "YOLOv12", "YOLOv12", arxiv="2502.12524"),
    E("lei2025yolov13", "01", "YOLOv13", "YOLOv13", arxiv="2506.17733"),
    E("robinson2025rfdetr", "01", "RF-DETR", "RF-DETR", arxiv="2511.09554"),
    E("rtdetrv4_2025", "01", "RT-DETRv4", "RT-DETRv4", arxiv="2510.25257"),
    E("deimv2_2025", "01", "DEIMv2", "DINOv3", arxiv="2509.20787"),
    E("huang2025deim", "01", "DEIM", "DEIM", arxiv="2412.04234"),
    E("peng2025dfine", "01", "D-FINE", "D-FINE", arxiv="2410.13842"),
    E("zhao2024rtdetr", "01", "RT-DETR", "Real-time Object Detection", arxiv="2304.08069"),
    E("zong2023codetr", "01", "Co-DETR", "Collaborative Hybrid", arxiv="2211.12860"),
    E("lyu2022rtmdet", "01", "RTMDet", "RTMDet", arxiv="2212.07784"),
    E("lin2017fpn", "01", "FPN", "Feature Pyramid Networks", arxiv="1612.03144"),
    E("liu2018panet", "01", "PANet", "Path Aggregation", arxiv="1803.01534"),
    E("yolo11_2024", "01", "YOLO11", note="software release, no paper: https://github.com/ultralytics/ultralytics"),
    # ---- state space models ----------------------------------------------
    E("gu2022s4", "02", "S4", "Structured State Spaces", arxiv="2111.00396"),
    E("smith2023s5", "02", "S5", "Simplified State Space", arxiv="2208.04933"),
    E("gu2023mamba", "02", "Mamba", "Mamba", arxiv="2312.00752"),
    E("dao2024mamba2", "02", "Mamba-2", "Transformers are SSMs", arxiv="2405.21060"),
    E("han2024mlla", "02", "MLLA", "Demystify Mamba", arxiv="2405.16605"),
    E("dao2023flash2", "02", "FlashAttention-2", "FlashAttention-2", arxiv="2307.08691"),
    E("zhu2024vim", "02", "Vim", "Vision Mamba", arxiv="2401.09417"),
    E("liu2024vmamba", "02", "VMamba", "VMamba", arxiv="2401.10166"),
    E("huang2024localmamba", "02", "LocalMamba", "LocalMamba", arxiv="2403.09338"),
    E("xiao2025spatialmamba", "02", "Spatial-Mamba", "Spatial-Mamba", arxiv="2410.15091"),
    E("damamba2025", "02", "DAMamba", "DAMamba", arxiv="2502.12627"),
    E("a2mamba2025", "02", "A2Mamba", "A2Mamba", arxiv="2507.16624"),
    E("prismamba2026", "02", "PRISMamba", "PRISMamba", arxiv="2602.04170"),
    E("vnct2026", "02", "VNCT", "Trapezoidal", arxiv="2607.03589"),
    E("graphscan2026", "02", "GraphScan", "GraphScan", arxiv="2605.11300"),
    E("cgspn2026", "02", "C-GSPN", "Parallel Sequence", arxiv="2606.00746"),
    E("sfmamba2026", "02", "SF-Mamba", "Mamba", arxiv="2603.16423"),
    E("yu2025mambaout", "02", "MambaOut", "MambaOut", arxiv="2405.07992"),
    E("hatamizadeh2025mambavision", "02", "MambaVision", "MambaVision", arxiv="2407.08083"),
    E("xie2024quadmamba", "02", "QuadMamba", "QuadMamba", search='ti:QuadMamba'),
    E("objm2025", "02", "ObjM", "Objectness", search='all:objectness AND all:mamba AND all:scan'),
    # ---- mamba detectors -------------------------------------------------
    E("wang2025mambayolo", "03", "Mamba-YOLO", "Mamba YOLO", arxiv="2406.05835"),
    E("mambapsa2026", "03", "MambaPSA", "MambaPSA", arxiv="2607.12681"),
    E("yolo12mambascan2026", "03", "YOLO12-MambaScan", "Mamba", arxiv="2609.13647"),
    E("scopemamba2026", "03", "ScopeMamba-YOLO", "ScopeMamba", arxiv="2609.10156"),
    E("hdmambayolo2026", "03", "HDMamba-YOLO", "Mamba", arxiv="2609.23061"),
    E("akcmamba2026", "03", "AKCMamba-YOLO", "AKCMamba", search='all:AKCMamba'),
    E("mambanextyolo2025", "03", "MambaNeXt-YOLO", "MambaNeXt", arxiv="2506.03654"),
    E("lcmamnet2026", "03", "LCMamNet", "LCMamNet", arxiv="2607.24184"),
    # ---- small objects ---------------------------------------------------
    E("sfdnet2026", "04", "SFDNet", "SFDNet", arxiv="2606.29029"),
    E("dernet2026", "04", "DERNet", "DERNet", arxiv="2606.23825"),
    E("o2deim2026", "04", "O2-DEIM", "Oriented", arxiv="2603.15497"),
    E("efsidetr2026", "04", "EFSI-DETR", "EFSI", arxiv="2601.18597"),
    E("tolf2026", "04", "TOLF", "Flow", arxiv="2601.00617"),
    E("domedetr2025", "04", "Dome-DETR", "Dome-DETR", arxiv="2505.05741"),
    E("yuan2026stripr", "04", "Strip-RCNN", "Strip R-CNN", arxiv="2501.03775"),
    E("uavdetr2025", "04", "UAV-DETR", "UAV-DETR", arxiv="2501.01855"),
    E("huang2024dqdetr", "04", "DQ-DETR", "DQ-DETR", arxiv="2404.03507"),
    E("wang2020aitod", "04", "AI-TOD", note="ICPR 2020; no open-access version found (title search returns a different 2024 paper)"),
    E("xu2022aitodv2", "04", "AI-TODv2-NWD", "Wasserstein", arxiv="2206.13996"),
    E("wang2021nwd", "04", "NWD", "Wasserstein", arxiv="2110.13389"),
    E("cheng2023soda", "04", "SODA", "Small Object Detection", arxiv="2207.14096"),
    E("yu2020tinyperson", "04", "TinyPerson-ScaleMatch", "Scale Match", arxiv="1912.10664"),
    E("zhu2021tph", "04", "TPH-YOLOv5", "TPH-YOLOv5", arxiv="2108.11539"),
    E("akyon2022sahi", "04", "SAHI", "Slicing Aided", arxiv="2202.06934"),
    E("wang2025hpsdetr", "04", "HPS-DETR", note="IEEE TGRS, not open access; the copy supplied is in 00_source_papers"),
    E("yang2026uavdet", "04", "UAVDet", note="CVIU, not open access; the copy supplied is in 00_source_papers"),
    # ---- RGB-thermal fusion ----------------------------------------------
    E("dong2025fusionmamba", "05", "Fusion-Mamba", "Fusion-Mamba", arxiv="2404.09146"),
    E("li2025cfmw", "05", "CFMW-preprint", "Fusion Mamba", arxiv="2404.16302"),
    E("como2024", "05", "COMO", "COMO", arxiv="2412.18076"),
    E("wavemamba2025", "05", "WaveMamba", "WaveMamba", arxiv="2507.18173"),
    E("ms2fusion2025", "05", "MS2Fusion", "Fusion", arxiv="2507.14643"),
    E("mambarefine2025", "05", "MambaRefine-YOLO", "MambaRefine", arxiv="2511.19134"),
    E("registerbridge2026", "05", "RegisterBridgeMM", "Register", arxiv="2608.04833"),
    E("cfgpnet2026", "05", "CFGPNet", "CFGPNet", arxiv="2608.06205"),
    E("remotedetmamba2024", "05", "RemoteDet-Mamba", "RemoteDet-Mamba", arxiv="2410.13532"),
    E("mambast2024", "05", "MambaST", "MambaST", arxiv="2408.01037"),
    E("dlrmamba2026", "05", "DLRMamba", "DLRMamba", arxiv="2603.06920"),
    E("msdfmamba2025", "05", "MSDF-Mamba", "MSDF", search='all:"MSDF-Mamba"'),
    E("cgssm2025", "05", "CG-SSM-misaligned-fusion", "Misaligned",
      search='ti:misaligned AND ti:multispectral AND ti:"state space"'),
    E("cbcswca2025", "05", "CBC-SWCA-position-shift", "Shift", arxiv="2502.09311"),
    E("qingyun2021cft", "05", "CFT", "Cross-Modality Fusion Transformer", arxiv="2111.00273"),
    E("shen2024icafusion", "05", "ICAFusion", "ICAFusion", arxiv="2308.07504"),
    E("chen2022proben", "05", "ProbEn", "Probabilistic Ensembling", arxiv="2104.02904"),
    # ---- video, tracking, streaming --------------------------------------
    E("zubic2024ssmevent", "06", "SSMs-for-event-cameras", "Event Cameras", arxiv="2402.15584"),
    E("gehrig2023rvt", "06", "RVT", "Recurrent Vision Transformers", arxiv="2212.05598"),
    E("tmambadet2026", "06", "TMambaDet", "TMambaDet", search='all:TMambaDet'),
    E("segu2025sambamotr", "06", "SambaMOTR", "Samba", arxiv="2410.01806"),
    E("gao2025motip", "06", "MOTIP", "ID Prediction", arxiv="2403.16848"),
    E("matr2025", "06", "MATR", "Track", arxiv="2509.21715"),
    E("spikingmot2026", "06", "SpikingMOT", "Spiking", arxiv="2607.19875"),
    E("smp2026", "06", "SAM3-mask-propagation", "Propagation", arxiv="2606.13033"),
    E("zhang2023motrv2", "06", "MOTRv2", "MOTRv2", arxiv="2211.09791"),
    E("cao2023ocsort", "06", "OC-SORT", "Observation-Centric", arxiv="2203.14360"),
    E("zhang2022bytetrack", "06", "ByteTrack", "ByteTrack", arxiv="2110.06864"),
    E("aharon2022botsort", "06", "BoT-SORT", "BoT-SORT", arxiv="2206.14651"),
    E("shim2025tracktrack", "06", "TrackTrack", url="https://openaccess.thecvf.com/content/CVPR2025/papers/"
      "Shim_Focusing_on_Tracks_for_Online_Multi-Object_Tracking_CVPR_2025_paper.pdf", year=2025,
      title="Focusing on Tracks for Online Multi-Object Tracking"),
    E("mambatrack2024", "06", "MambaTrack", "MambaTrack", arxiv="2408.09178"),
    E("trackssm2024", "06", "TrackSSM", "TrackSSM", arxiv="2409.00487"),
    E("li2020streaming", "06", "Towards-Streaming-Perception", "Streaming Perception", arxiv="2005.10420"),
    E("yang2022streamyolo", "06", "StreamYOLO", "Streaming Perception", arxiv="2203.12338"),
    E("li2023longshortnet", "06", "LongShortNet", "LongShortNet", arxiv="2210.15518"),
    E("he2023damostreamnet", "06", "DAMO-StreamNet", "DAMO-StreamNet", arxiv="2303.17144"),
    E("luiten2021hota", "06", "HOTA", "HOTA", arxiv="2009.07736"),
    E("ristani2016idf1", "06", "IDF1-performance-measures", "Performance Measures", arxiv="1609.01775"),
    E("msenet2024", "06", "MSENet-UVT-VOD2024", "Video", arxiv="2410.12143"),
    E("msgnet2025", "06", "MSGNet", "Video", arxiv="2504.11779"),
    E("einet2023", "06", "EINet-VT-VOD50", "Erasure", arxiv="2308.01630"),
    E("dchnet2026", "06", "DCHNet-DVT-VOD1000", "Video", arxiv="2607.08191"),
    E("scdt2026", "06", "SCDT", "Tracking", arxiv="2607.24701"),
    E("hgttrack2024", "06", "HGT-Track-VT-Tiny-MOT", "Tiny", arxiv="2412.10861"),
    E("ballas2016convgru", "06", "ConvGRU", "Delving Deeper", arxiv="1511.06432"),
    E("williams1990tbptt", "06", "TBPTT", note="Neural Computation 2(4), 1990; not open access"),
    # ---- datasets ----------------------------------------------------------
    E("lin2014coco", "07", "COCO", "Common Objects in Context", arxiv="1405.0312"),
    E("singh2024cocorem", "07", "COCO-ReM", "COCO", arxiv="2403.18819"),
    E("gupta2019lvis", "07", "LVIS", "LVIS", arxiv="1908.03195"),
    E("shao2019objects365", "07", "Objects365", url="https://openaccess.thecvf.com/content_ICCV_2019/papers/"
      "Shao_Objects365_A_Large-Scale_High-Quality_Dataset_for_Object_Detection_ICCV_2019_paper.pdf", year=2019,
      title="Objects365: A Large-Scale, High-Quality Dataset for Object Detection"),
    E("kuznetsova2020openimages", "07", "OpenImages-V4", "Open Images", arxiv="1811.00982"),
    E("everingham2010voc", "07", "PASCAL-VOC", url="http://host.robots.ox.ac.uk/pascal/VOC/pubs/everingham10.pdf", year=2010,
      title="The PASCAL Visual Object Classes (VOC) Challenge"),
    E("russakovsky2015imagenet", "07", "ImageNet-ILSVRC", "ImageNet", arxiv="1409.0575"),
    E("zhu2021visdrone", "07", "VisDrone", "Drones", arxiv="2001.06303"),
    E("du2018uavdt", "07", "UAVDT", "Unmanned Aerial Vehicle", arxiv="1804.00518"),
    E("xia2018dota", "07", "DOTA", "DOTA", arxiv="1711.10398"),
    E("ding2021dotav2", "07", "DOTA-v2", "Aerial Images", arxiv="2102.12219"),
    E("lam2018xview", "07", "xView", "xView", arxiv="1802.07856"),
    E("varga2022seadronessee", "07", "SeaDronesSee", "SeaDronesSee", arxiv="2105.01922"),
    E("xsvid2024", "07", "XS-VID", "XS-VID", arxiv="2407.18137"),
    E("tinyset2026", "07", "TinySet-9M", "Tiny", arxiv="2604.02773"),
    E("jia2021llvip", "07", "LLVIP", "LLVIP", arxiv="2108.10831"),
    E("liu2022tardal", "07", "M3FD-TarDAL", "Dual Adversarial", arxiv="2203.16220"),
    E("hwang2015kaist", "07", "KAIST", url="https://openaccess.thecvf.com/content_cvpr_2015/papers/"
      "Hwang_Multispectral_Pedestrian_Detection_2015_CVPR_paper.pdf", year=2015,
      title="Multispectral Pedestrian Detection: Benchmark Dataset and Baseline"),
    E("li2018kaistsanitized", "07", "KAIST-sanitized", "Multispectral Pedestrian", arxiv="1808.04818"),
    E("zhang2020cfr", "07", "FLIR-aligned-CFR", "Fuse-and-Refine", arxiv="2009.12664"),
    E("sun2022dronevehicle", "07", "DroneVehicle", "Drone-based", arxiv="2003.02437"),
    E("razakarivony2016vedai", "07", "VEDAI", note="J. Visual Communication and Image Representation 2016; not on arXiv; dataset page: https://downloads.greyc.fr/vedai/"),
    E("gonzalez2016cvc14", "07", "CVC-14", note="open access but the publisher blocks scripted downloads; "
      "download from https://www.mdpi.com/1424-8220/16/6/820"),
    E("ying2025rgbttiny", "07", "RGBT-Tiny", "Tiny Object", arxiv="2406.14482"),
    E("vtmot2024", "07", "VT-MOT", "Multiple Object Tracking", arxiv="2408.00969"),
    E("m3ot2025", "07", "M3OT", note="open access but the publisher blocks scripted downloads; "
      "download from https://www.nature.com/articles/s41597-025-06204-0"),
    E("mmot2025", "07", "MMOT", "Multispectral", arxiv="2510.12565"),
    E("drgbt2026", "07", "DRGBT-1K", "RGBT", arxiv="2607.19772"),
    E("milan2016mot16", "07", "MOT16-MOT17", "MOT16", arxiv="1603.00831"),
    E("dendorfer2020mot20", "07", "MOT20", "MOT20", arxiv="2003.09003"),
    E("sun2022dancetrack", "07", "DanceTrack", "DanceTrack", arxiv="2111.14690"),
    E("cui2023sportsmot", "07", "SportsMOT", "SportsMOT", arxiv="2304.05170"),
    E("yu2020bdd100k", "07", "BDD100K", "BDD100K", arxiv="1805.04687"),
    E("dave2020tao", "07", "TAO", "Tracking Any Object", arxiv="2005.10356"),
    E("fan2019lasot", "07", "LaSOT", "LaSOT", arxiv="1809.07845"),
    E("corona2021meva", "07", "MEVA", "MEVA", arxiv="2012.00914"),
    E("robicheaux2025rf100vl", "07", "RF100-VL", "Roboflow100", arxiv="2505.20612"),
    # ---- foundation models -------------------------------------------------
    E("oquab2024dinov2", "08", "DINOv2", "DINOv2", arxiv="2304.07193"),
    E("simeoni2025dinov3", "08", "DINOv3", "DINOv3", arxiv="2508.10104"),
    E("sam3_2025", "08", "SAM3", "SAM 3", arxiv="2511.16719"),
    E("wang2025yoloe", "08", "YOLOE", "YOLOE", arxiv="2503.07465"),
    # ---- other methods -----------------------------------------------------
    E("chen2023fasternet", "09", "FasterNet", "FLOPS", arxiv="2303.03667"),
    E("liu2023efficientvit", "09", "EfficientViT", "EfficientViT", arxiv="2305.07027"),
    E("liu2023dysample", "09", "DySample", "Learning to Upsample", arxiv="2308.15085"),
    E("kang2024asfyolo", "09", "ASF-YOLO", "ASF-YOLO", arxiv="2312.06458"),
    E("zhang2023inneriou", "09", "Inner-IoU", "Inner-IoU", arxiv="2311.02877"),
    E("ma2023mpdiou", "09", "MPDIoU", "MPDIoU", arxiv="2307.07662"),
    E("ho2020ddpm", "09", "DDPM", "Denoising Diffusion Probabilistic", arxiv="2006.11239"),
    E("song2021ddim", "09", "DDIM", "Denoising Diffusion Implicit", arxiv="2010.02502"),
    E("ozdenizci2023weatherdiff", "09", "WeatherDiff", "Adverse Weather", arxiv="2207.14626"),
    # ---- law and ethics ----------------------------------------------------
    E("euaiact2024", "10", "EU-AI-Act-Regulation-2024-1689",
      url="https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=CELEX:32024R1689", year=2024,
      title="Regulation (EU) 2024/1689 (Artificial Intelligence Act)"),
    E("gdpr2016", "10", "GDPR-Regulation-2016-679",
      url="https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=CELEX:32016R0679", year=2016,
      title="Regulation (EU) 2016/679 (General Data Protection Regulation)"),
    E("peng2021stewardship", "10", "Dataset-stewardship-1000-papers", "Stewardship", arxiv="2108.02922"),
    E("dpdp2023", "10", "India-DPDP-Act-2023", note="official text: https://www.meity.gov.in (Digital Personal Data Protection Act, 2023)"),
    E("dpdprules2025", "10", "India-DPDP-Rules-2025", note="notified November 2025; official text on meity.gov.in and egazette.gov.in"),
    E("puttaswamy2017", "10", "Puttaswamy-2017", note="Supreme Court of India judgment, (2017) 10 SCC 1"),
    E("poi2011", "10", "Person-of-Interest", note="television series, not a paper"),
    # ---- software and documentation (listed, not downloaded) ---------------
    E("mambassm_repo", "02", "mamba_ssm", note="software: https://github.com/state-spaces/mamba"),
    E("mambapy_repo", "02", "mamba.py", note="software: https://github.com/alxndrTL/mamba.py"),
    E("trackeval_repo", "06", "TrackEval", note="software: https://github.com/JonathonLuiten/TrackEval"),
    E("fastercocoeval_repo", "07", "faster-coco-eval", note="software: https://github.com/MiXaiLL76/faster_coco_eval"),
    E("pyav_repo", "06", "PyAV", note="software: https://github.com/PyAV-Org/PyAV"),
    E("gradio_repo", "06", "Gradio", note="software: https://github.com/gradio-app/gradio"),
    E("fastrtc_repo", "06", "FastRTC", note="software: https://github.com/gradio-app/fastrtc"),
    E("roboflowtrackers_repo", "06", "trackers", note="software: https://github.com/roboflow/trackers"),
    E("deepstream_docs", "06", "DeepStream", note="documentation: https://docs.nvidia.com/metropolis/deepstream"),
]


BROWSER_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
              "(KHTML, like Gecko) Chrome/130.0 Safari/537.36")


def http_get(url, timeout=120, retries=4, ua=UA):
    """GET with back-off on rate limiting (HTTP 429) and transient server errors."""
    req = urllib.request.Request(url, headers={"User-Agent": ua, "Accept": "application/pdf,*/*"})
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except urllib.error.HTTPError as exc:
            if exc.code not in (429, 500, 502, 503, 504) or attempt == retries:
                raise
        except (urllib.error.URLError, TimeoutError):
            if attempt == retries:
                raise
        time.sleep(20 * 2 ** attempt)


def pdf_first_page(data):
    """Return (title, text) of the first page: the title is the largest-font line."""
    import pymupdf
    doc = pymupdf.open(stream=data, filetype="pdf")
    page = doc[0]
    text = " ".join(page.get_text().split())
    spans = [s for b in page.get_text("dict")["blocks"] for l in b.get("lines", [])
             if abs(l["dir"][0]) > 0.9  # skip the rotated arXiv stamp in the margin
             for s in l["spans"] if s["text"].strip() and s["bbox"][1] < page.rect.height * 0.5
             and not s["text"].strip().lower().startswith("arxiv:")]
    if not spans:
        return "", text
    biggest = max(s["size"] for s in spans)
    title = " ".join(s["text"].strip() for s in spans if s["size"] >= biggest - 0.6)
    return " ".join(title.split())[:220], text


def arxiv_query(params):
    data = http_get("https://export.arxiv.org/api/query?" + urllib.parse.urlencode(params))
    time.sleep(DELAY)
    root = ET.fromstring(data)
    out = []
    for e in root.findall("a:entry", ATOM):
        title = " ".join((e.findtext("a:title", "", ATOM) or "").split())
        aid = (e.findtext("a:id", "", ATOM) or "").rsplit("/abs/", 1)[-1]
        if not title or title == "Error":
            continue
        out.append((re.sub(r"v\d+$", "", aid), title))
    return out


def norm(text):
    """Lower-case, undo typographic ligatures (fi, fl) and collapse whitespace."""
    import unicodedata
    return " ".join(unicodedata.normalize("NFKC", text).replace("­", "").lower().split())


def matches(title, check):
    return check is None or norm(check) in norm(title)


def fetch_arxiv(aid, check):
    """Download an arXiv PDF and confirm from its first page that it is the expected paper."""
    data = http_get(f"https://export.arxiv.org/pdf/{aid}")
    time.sleep(DELAY)
    if not data.startswith(b"%PDF"):
        raise ValueError("response is not a PDF")
    title, text = pdf_first_page(data)
    return data, title, matches(text[:3000], check)


def resolve(p):
    """Return (arxiv_id, pdf_bytes, title, note) or (None, None, "", reason)."""
    note = ""
    if p["arxiv"]:
        data, title, ok = fetch_arxiv(p["arxiv"], p["check"])
        if ok:
            return p["arxiv"], data, title, ""
        note = f"listed id {p['arxiv']} is a different paper ({title[:80]})"
    query = p["search"] or (f'ti:"{p["check"]}"' if p["check"] else None)
    if query:
        for aid, api_title in arxiv_query({"search_query": query, "max_results": 5}):
            if matches(api_title, p["check"]) and aid != p["arxiv"]:
                data, title, ok = fetch_arxiv(aid, p["check"])
                if ok:
                    return aid, data, title or api_title, (note + "; found by title search").strip("; ")
    return None, None, "", note or "no arXiv match"


def year_of(aid, fallback):
    m = re.match(r"(\d{2})(\d{2})\.", aid or "")
    return 2000 + int(m.group(1)) if m else (fallback or "")


def save(path, data):
    if not data.startswith(b"%PDF"):
        raise ValueError("response is not a PDF")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(data)


def refresh_titles(rows):
    """Replace titles read from PDFs with the clean titles held by arXiv (one call per 50 papers)."""
    ids = {r["source"].rsplit("/abs/", 1)[1]: r for r in rows if "/abs/" in r["source"]}
    keys = list(ids)
    for i in range(0, len(keys), 50):
        try:
            for aid, title in arxiv_query({"id_list": ",".join(keys[i:i + 50]), "max_results": 50}):
                if aid in ids:
                    ids[aid]["title"] = title
        except Exception as exc:  # titles from the PDFs remain in place
            print("title refresh skipped:", exc)


HEADINGS = {
    "00_source_papers": "Source papers supplied with the topic",
    "01_detectors": "Real-time detectors",
    "02_state_space_models": "State-space models and vision SSMs",
    "03_mamba_detectors": "Mamba-based detectors",
    "04_small_objects": "Small and tiny object detection",
    "05_rgbt_fusion": "Visible-thermal (RGB-T) fusion",
    "06_video_tracking_streaming": "Video detection, tracking and streaming",
    "07_datasets": "Datasets and benchmarks",
    "08_foundation_models": "Foundation models",
    "09_other_methods": "Other methods (blocks, losses, diffusion)",
    "10_law_and_ethics": "Law and ethics",
}


def write_readme(rows):
    """Write papers/README.md: one table per category, linked to the local PDFs."""
    have = sum(r["status"] == "downloaded" for r in rows)
    out = ["# Papers", "",
           "Every work cited in the research dossier (`report/CFFM-Net_Research_Dossier.pdf`), "
           "grouped as in the report. Open-access papers are stored here as PDFs; the rest are "
           "listed with the reason and a link. `index.csv` holds the same information in "
           "machine-readable form, keyed by the citation keys of `report/references.tex`.", "",
           f"{have} of {len(rows)} cited works are stored as PDFs. To refresh, run "
           "`python papers/download_papers.py` (files already present are skipped). arXiv "
           "identifiers were confirmed against each PDF's first page.", ""]
    src = sorted(f for f in os.listdir(os.path.join(ROOT, "00_source_papers")) if f.endswith(".pdf"))
    out += [f"## {HEADINGS['00_source_papers']}", ""]
    out += [f"- [{f[:-4]}](00_source_papers/{urllib.parse.quote(f)})" for f in src] + [""]
    for cat, heading in HEADINGS.items():
        group = [r for r in rows if r["category"] == cat]
        if not group:
            continue
        out += [f"## {heading}", "", "| Key | Title | File | Source or note |", "|---|---|---|---|"]
        for r in sorted(group, key=lambda r: (r["status"] != "downloaded", r["file"] or r["key"])):
            title = (r["title"] or r["name"]).replace("|", "/")
            link = f"[pdf]({urllib.parse.quote(r['file'])})" if r["file"] else "not stored"
            note = r["source"] if r["status"] == "downloaded" else (r["note"] or r["status"])
            out.append(f"| `{r['key']}` | {title} | {link} | {note.replace('|', '/')} |")
        out.append("")
    with open(os.path.join(ROOT, "README.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(out))


def main():
    rows = []
    for i, p in enumerate(PAPERS, 1):
        folder = os.path.join(ROOT, CATS[p["cat"]])
        row = dict(key=p["key"], category=CATS[p["cat"]], name=p["name"], status="", file="",
                   title="", source="", note=p["note"] or "")
        try:
            if p["note"] and not (p["arxiv"] or p["url"] or p["search"]):
                row["status"] = "not downloaded"
            elif p["url"]:
                fname = f"{p['year']}_{p['name']}.pdf"
                path = os.path.join(folder, fname)
                if not os.path.exists(path):
                    save(path, http_get(p["url"], retries=1, ua=BROWSER_UA))
                    time.sleep(1.0)
                with open(path, "rb") as fh:
                    title, _ = pdf_first_page(fh.read())
                row.update(status="downloaded", file=os.path.relpath(path, ROOT).replace(os.sep, "/"),
                           source=p["url"], title=p["title"] or title)
            else:
                done = [f for f in os.listdir(folder) if f.startswith(f"{p['year'] or ''}") and
                        f"_{p['name']}_" in f] if os.path.isdir(folder) else []
                if done:  # already fetched on an earlier run
                    path = os.path.join(folder, done[0])
                    with open(path, "rb") as fh:
                        title, _ = pdf_first_page(fh.read())
                    aid = done[0].rsplit("_", 1)[-1][:-4]
                    row.update(status="downloaded", file=os.path.relpath(path, ROOT).replace(os.sep, "/"),
                               title=title, source=f"https://arxiv.org/abs/{aid}")
                else:
                    aid, data, title, note = resolve(p)
                    if aid is None:
                        row.update(status="not found", note=(note + "; " + row["note"]).strip("; "))
                    else:
                        fname = f"{year_of(aid, p['year'])}_{p['name']}_{aid.replace('/', '-')}.pdf"
                        path = os.path.join(folder, fname)
                        save(path, data)
                        row.update(status="downloaded", file=os.path.relpath(path, ROOT).replace(os.sep, "/"),
                                   title=title, source=f"https://arxiv.org/abs/{aid}",
                                   note=(note + "; " + row["note"]).strip("; "))
        except Exception as exc:  # keep going; the index records the failure
            row.update(status="failed", note=f"{type(exc).__name__}: {exc}")
        rows.append(row)
        print(f"[{i:3d}/{len(PAPERS)}] {row['status']:14s} {p['key']:28s} {row['title'][:60]}", flush=True)

    refresh_titles(rows)
    with open(os.path.join(ROOT, "index.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    write_readme(rows)
    print("done:", sum(r["status"] == "downloaded" for r in rows), "of", len(rows), "downloaded")


if __name__ == "__main__":
    sys.exit(main())
