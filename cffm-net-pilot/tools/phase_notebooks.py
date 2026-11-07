"""Phase 2, 3 and 4 notebooks, one per phase; cells and helpers shared with build_notebooks.py."""
from build_notebooks import CLEANUP, SETUP, code, figure, header, md, write

KAIST_SETUP = '''
from pathlib import Path
import json, time
import cv2, numpy as np, pandas as pd, torch
from cffm import kaist, video, faults, hud, showcase as sc
from cffm.train import _load

INPUT = [Path("/kaggle/input")] if ON_KAGGLE else [env.project / "data" / "raw"]
IMG_ROOT, ANN_ROOT = kaist.find_roots(INPUT)
dev = "cuda:0" if torch.cuda.is_available() else "cpu"
out = env.project / "results" / "PHASE"
out.mkdir(parents=True, exist_ok=True)
print("KAIST images:", IMG_ROOT, "| annotations:", ANN_ROOT, "| device:", dev)
'''


def nb20():
    write("notebooks/phase2_video/20_phase2_video.ipynb", [
        header("20 · Phase 2: video, memory and tracking on KAIST",
               "Phase 2 turns the still-image detector into a video analyser. It trains CFFM-Net and the concat "
               "baseline on KAIST, the standard visible-thermal pedestrian benchmark (95,000 aligned frames at "
               "20 fps), adds the Temporal State Memory and trains it on clips with the detector frozen, then "
               "measures what the memory buys: log-average miss rate on the standard test set, clean and while a "
               "camera is failing, and MOTA and IDF1 when tracking with ByteTrack.",
               "`cffm-net-pilot`, `adlteam/kaist-dataset` (frames) and `arnb007/annotation-of-kaist-dataset` (vbb)",
               "about 9 hours on 2 x T4; every step resumes after a restart",
               "`results/phase2/`: miss-rate and tracking tables, memory weights, failure boards and a clip"),
        SETUP,
        code(KAIST_SETUP.replace("PHASE", "phase2")),
        md("## 1. The benchmark\n\nTraining uses sets 00 to 05 and testing sets 06 to 11, the standard split. Stills "
           "for detector training are every 4th training frame; the still-image test set is the standard every "
           "20th frame. The headline metric is the log-average miss rate over 10^-2 to 10^0 false positives per "
           "image on the *reasonable* subset (people at least 55 px tall, at most partly occluded): lower is "
           "better."),
        code('''
ds = plan["datasets"]["kaist"]
kdata = env.data / "kaist"
if not (kdata / "data_paired.yaml").exists():
    counts = kaist.convert(IMG_ROOT, ANN_ROOT, kdata, train_stride=ds["train_stride"], test_stride=ds["test_stride"])
    print("converted", counts)
train_seqs = kaist.sequences(IMG_ROOT, ANN_ROOT, kaist.TRAIN_SETS)
test_seqs = kaist.sequences(IMG_ROOT, ANN_ROOT, kaist.TEST_SETS)
rows = [{"split": n, "sequences": len(s), "frames": sum(len(q.frames) for q in s),
         "person boxes": sum(1 for q in s for f in q.boxes for b in f if b.label == "person"),
         "reasonable": sum(1 for q in s for f in q.boxes for b in f if b.reasonable),
         "night sequences": sum(not q.daytime for q in s)} for n, s in (("train", train_seqs), ("test", test_seqs))]
pd.DataFrame(rows)
'''),
        md("## 2. Train the detectors\n\nCFFM-Net v2 and the concat baseline with the same recipe: COCO-pretrained "
           "YOLO26-n backbones, modality dropout 0.15, 25 epochs, two GPUs. Finished runs are reused."),
        code('''
for s in pipeline.runs_for(plan, "20"):
    pipeline.train_from_spec(s, env)
ckpts = pipeline.find_checkpoints(env)
models = {"concat": _load(ckpts["kaist_concat_md"]).to(dev), "cffm": _load(ckpts["kaist_cffm_v2"]).to(dev)}
'''),
        md("## 3. Still-image results\n\nEach test frame on its own, no memory: the KAIST table every paper reports."),
        code('''
cache = out / "stills.json"
if cache.exists():
    stills = json.loads(cache.read_text())
else:
    stills = {n: video.evaluate_sequences(m, test_seqs, stride=20, memory=False, device=dev) for n, m in models.items()}
    cache.write_text(json.dumps(stills, indent=2))
(pd.DataFrame(stills).T[["MR_all", "MR_day", "MR_night", "AP50_all"]] * 100).round(2)
'''),
        md("## 4. Train the Temporal State Memory\n\nOne memory block per fusion level, zero-initialised so the "
           "detector starts exactly as trained. The detector is frozen; only the memory learns, by backpropagation "
           "through six frames at 10 fps with the detection loss on every frame. Half the clips lose or corrupt a "
           "camera part-way through, so the memory learns to carry what it saw across the failure."),
        figure("fig_tsm"),
        code('''
cffm = models["cffm"]
mem_path = out / "memory.pt"
cffm.attach_memory()
saved = [m for m in [mem_path, *Path("/kaggle/input").glob("**/results/phase2/memory.pt")] if m.exists()]
saved = [m for m in saved if all(torch.isfinite(v).all() for v in torch.load(m, map_location="cpu").values())]
if saved:
    cffm.memory.load_state_dict(torch.load(saved[0], map_location=dev))
    print("memory loaded from", saved[0])
else:
    hist = video.train_memory(cffm, train_seqs, epochs=3, T=6, stride=2, batch=4, lr=5e-4, workers=4, device=dev)
    torch.save(cffm.memory.state_dict(), mem_path)
    pd.DataFrame(hist, columns=["step", "loss"]).plot(x="step", y="loss", figsize=(7, 3), legend=False,
                                                       title="memory training loss")
print("memory parameters:", sum(p.numel() for p in cffm.memory.parameters()))
'''),
        md("## 5. What the memory buys\n\nThe test sequences run causally at 10 fps, carrying state frame to frame, "
           "and are scored on the standard every-20th-frame grid. Each failure repeats: 4 s healthy, then 4 s "
           "failed. *Failed* scores only frames inside a failure; *flagged* means the stream monitor reported the "
           "dropped camera. The detector weights are identical in every row; only the memory differs."),
        code('''
SCHEDULES = {"clean": None,
             "thermal lost": video.periodic("thermal_drop", 3, on=80, off=80),
             "visible dark": video.periodic("visible_dark", 3, on=80, off=80),
             "thermal noise": video.periodic("thermal_noise", 3, on=80, off=80)}
cache = out / "memory.json"
mem = json.loads(cache.read_text()) if cache.exists() else {}
for name, sched in SCHEDULES.items():
    for flagged in ([False, True] if name == "thermal lost" else [False]):
        for use in (False, True):
            key = f"{name}{' (flagged)' if flagged else ''} | {'memory' if use else 'image'}"
            if key in mem:
                continue
            t0 = time.time()
            mem[key] = video.evaluate_sequences(cffm, test_seqs, stride=2, memory=use, schedule=sched,
                                                flags=flagged, device=dev)
            cache.write_text(json.dumps(mem, indent=2))
            print(f"{key:40s} MR {mem[key]['MR_all'] * 100:5.2f}  ({(time.time() - t0) / 60:.1f} min)")
rows = []
for k in sorted({k.split(" | ")[0] for k in mem}):
    a, b = mem[f"{k} | image"], mem[f"{k} | memory"]
    for col in ("MR_all", "MR_failed", "MR_night"):
        if col in a:
            rows.append({"condition": k, "metric": col, "image": a[col] * 100, "memory": b[col] * 100,
                         "change (pts, lower is better)": (b[col] - a[col]) * 100})
pd.DataFrame(rows).round(2)
'''),
        md("## 6. Tracking\n\nByteTrack on every frame of four test sequences (two by day, two by night) at 20 fps, "
           "with and without memory, clean and with the thermal camera failing. MOTA and IDF1 against KAIST's "
           "track ids."),
        code('''
cache = out / "tracking.json"
trk = json.loads(cache.read_text()) if cache.exists() else {}
wanted = ("set06/V000", "set08/V000", "set09/V000", "set10/V001")
pick = [s for s in test_seqs if s.name in wanted] or test_seqs[:4]
for name in ("clean", "thermal lost"):
    for use in (False, True):
        key = f"{name} | {'memory' if use else 'image'}"
        if key in trk:
            continue
        per = []
        for s in pick:
            tracks = video.track_sequence(cffm, s, memory=use, schedule=SCHEDULES[name], device=dev, flags=True)
            per.append({"seq": s.name, **video.mot_metrics(tracks, s)})
        trk[key] = per
        cache.write_text(json.dumps(trk, indent=2))
summary = {k: pd.DataFrame(v)[["mota", "idf1", "num_switches"]].mean().to_dict() for k, v in trk.items()}
pd.DataFrame(summary).T.round(3)
'''),
        md("## 7. The Machine's view of a failure\n\nA night sequence at the moment the thermal camera is cut. Top "
           "of each board: image mode; bottom: with memory."),
        code('''
seq = next(s for s in test_seqs if not s.daytime)
active = video.periodic("thermal_drop", 3, on=80, off=80)[0]
boards = []
cffm.reset_memory()
for i in range(40, 240, 2):
    x = video.to_tensor(video.read_frame(seq.frames[i]))[None].to(dev)
    if active(i):
        x = faults.apply(x, "thermal_drop", 3)
    img = (x[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    tiles = []
    for use in (False, True):
        cffm.memory_on = use
        with torch.no_grad():
            d = video.detect(cffm, x, conf=0.3)[0]
        tiles.append(hud.render(img, d, {0: "person"}, frame=i, panel_w=640,
                                title="CFFM-NET + MEMORY" if use else "CFFM-NET",
                                subtitle="THERMAL LOST" if active(i) else "BOTH CAMERAS", panels=("visible", "thermal")))
    boards.append(hud.grid(tiles, cols=1))
hud.save_video(boards, out / "failure_memory.mp4", fps=10, width=1280)
sc.show(boards[60])
'''),
        code('''
(out / "phase2_summary.json").write_text(json.dumps({"stills": stills, "memory": mem, "tracking": summary},
                                                    indent=2, default=float))
print("saved", sorted(p.name for p in out.iterdir()))
'''),
        CLEANUP,
    ])


def nb30():
    write("notebooks/phase3_live/30_phase3_live.ipynb", [
        header("30 · Phase 3: live paired cameras",
               "Phase 3 runs CFFM-Net with its memory on live streams. Each camera has its own capture thread and a "
               "one-frame queue, so the model always sees the present. A stream monitor watches arrival times and "
               "pixels and raises the sensor flags, and the real time between frames sets the memory's step. "
               "Recorded KAIST test sequences are replayed in real time as cameras, so every run is repeatable; "
               "the same runtime takes webcams and RTSP URLs through `stream.DeviceSource`.",
               "`cffm-net-pilot`, KAIST frames and annotations, and the output of notebook 20",
               "about 40 minutes on one T4",
               "`results/phase3/`: soak and fault tables, latency plot, a recorded live session"),
        SETUP,
        code(KAIST_SETUP.replace("PHASE", "phase3") + '''
from cffm import stream
ckpts = pipeline.find_checkpoints(env)
name = "kaist_cffm_v2" if "kaist_cffm_v2" in ckpts else "llvip_cffm_v2_e80"
model = _load(ckpts[name]).to(dev).eval()
model.attach_memory()
mem = next(iter(sorted(Path("/kaggle/input" if ON_KAGGLE else env.project).glob("**/results/phase2/memory.pt"))), None)
if mem is not None:
    model.memory.load_state_dict(torch.load(mem, map_location=dev))
print("model:", name, "| memory:", mem or "untrained (identity)")
test_seqs = kaist.sequences(IMG_ROOT, ANN_ROOT, kaist.TEST_SETS)
frames = [f for s in test_seqs for f in s.frames]
irs = [kaist.lwir(f) for f in frames]
gt = {i: b for i, b in enumerate(b for s in test_seqs for b in s.boxes)}


def cameras(fps, vis_faults=None, ir_faults=None):
    t0 = time.perf_counter() + 1.0
    v = stream.Camera("visible", lambda: stream.ReplaySource(frames, fps=fps, t0=t0, faults=vis_faults))
    i = stream.Camera("thermal", lambda: stream.ReplaySource(irs, fps=fps, grey=True, t0=t0, faults=ir_faults),
                      grey=True)
    return v, i, t0
'''),
        md("## 1. Soak test\n\nThe exit test: paired 640 x 512 cameras at 30 fps for ten minutes, with no growth in "
           "delay or memory. Frames the model cannot keep up with are dropped at capture, never queued."),
        code('''
cache = out / "soak.json"
if cache.exists():
    soak = json.loads(cache.read_text())
else:
    soak = {}
    for fps, secs in ((30, 600), (20, 120)):
        v, i, t0 = cameras(fps)
        rt = stream.Runtime(model, v, i, fps=fps, device=dev)
        rec = rt.run(secs)
        rt.close()
        s = stream.summary(rec)
        s["capture_drop_rate"] = v.q.dropped / max(v.q.put_count, 1)
        soak[f"{fps} fps"] = s
        e2e = pd.Series([r["e2e_ms"] for r in rec], index=[r["t"] - t0 for r in rec])
        e2e.rolling(30).median().plot(figsize=(8, 3), label=f"{fps} fps", legend=True,
                                      title="end-to-end delay (ms), rolling median")
    cache.write_text(json.dumps(soak, indent=2))
pd.DataFrame(soak).T.round(2)
'''),
        md("## 2. Faults on the link\n\nNearly two minutes at 20 fps with a scripted sequence of failures: the "
           "thermal cable pulled for 6 s, the visible camera frozen for 6 s, the thermal stream stalled for 3 s, "
           "10 % random drops and 60 ms of extra delay on thermal. The monitor must flag each fault quickly, clear "
           "it when the camera returns, and raise no false alarms. The same run without the monitor shows what the "
           "flags are worth."),
        code('''
cache = out / "faults.json"
SCRIPT = {"thermal unplugged": ("ir", 20, 26), "visible frozen": ("vis", 50, 56), "thermal stalled": ("ir", 80, 83)}
if cache.exists():
    fres = json.loads(cache.read_text())
else:
    fres = {}
    for monitor in (True, False):
        ir_f = stream.StreamFaults(unplugs=[(20, 26)], stalls=[(80, 83)], drop_p=0.1, delay_s=0.06)
        vis_f = stream.StreamFaults(freezes=[(50, 56)], drop_p=0.1)
        v, i, t0 = cameras(20, vis_f, ir_f)
        rt = stream.Runtime(model, v, i, fps=20, device=dev, use_monitor=monitor)
        rec = rt.run(110)
        rt.close()
        dets = [r["dets"] for r in rec if r["vis_idx"] >= 0]
        boxes = [gt[r["vis_idx"] % len(frames)] for r in rec if r["vis_idx"] >= 0]
        fres["monitor on" if monitor else "monitor off"] = {
            "summary": stream.summary(rec), "reconnects": i.reconnects, "MR_all": kaist.miss_rate(dets, boxes),
            "faults": {k: stream.detection_delay(rec, a, b, camera=c, start=t0) for k, (c, a, b) in SCRIPT.items()}}
    cache.write_text(json.dumps(fres, indent=2, default=float))
pd.DataFrame(fres["monitor on"]["faults"]).T.round(3)
'''),
        code('''
pd.DataFrame({k: {"miss rate (%)": v["MR_all"] * 100, "fps": v["summary"]["fps"],
                  "e2e p95 (ms)": v["summary"]["e2e_p95"], "camera reconnects": v["reconnects"]}
              for k, v in fres.items()}).T.round(2)
'''),
        md("## 3. The live board\n\nForty seconds of the live system as the operator sees it: both cameras, every "
           "subject tracked and numbered, the health of each link and the delay of each frame. The thermal cable is "
           "pulled at 15 s and plugged back at 21 s."),
        code('''
boards = []


def draw(vis, ir, rec):
    if len(boards) >= 800 or vis is None or ir is None:
        return
    img = np.concatenate([vis.image, cv2.resize(ir.image, vis.image.shape[1::-1])[..., None]], 2)
    tr = np.asarray(rec["tracks"]).reshape(-1, 8) if len(rec["tracks"]) else np.zeros((0, 8))
    det = np.column_stack([tr[:, :4], tr[:, 5], tr[:, 6]]) if len(tr) else np.zeros((0, 6))
    status = f"VISIBLE {rec['vis'].upper()}   THERMAL {rec['ir'].upper()}"
    boards.append(hud.render(img, det, {0: "person"}, ids=tr[:, 4].astype(int).tolist() if len(tr) else None,
                             latency_ms=rec["e2e_ms"], panel_w=640, title="CFFM-NET LIVE", subtitle=status,
                             panels=("visible", "thermal")))


v, i, t0 = cameras(20, ir_faults=stream.StreamFaults(unplugs=[(15, 21)]))
rt = stream.Runtime(model, v, i, fps=20, device=dev, render=draw, conf=0.3)
rt.run(40)
rt.close()
hud.save_video(boards, out / "live_session.mp4", fps=20, width=1280)
sc.show(boards[len(boards) // 2])
'''),
        code('''
(out / "phase3_summary.json").write_text(json.dumps({"soak": soak, "faults": fres}, indent=2, default=float))
print("saved", sorted(p.name for p in out.iterdir()))
'''),
    ])


def nb40():
    write("notebooks/phase4_multicamera/40_phase4_multicamera.ipynb", [
        header("40 · Phase 4: anonymous multi-camera analytics",
               "Phase 4 joins calibrated cameras on one ground plane. WILDTRACK gives seven synchronised, calibrated "
               "views of a busy square, with every person located on a 2.5 cm grid. A person detector fine-tuned on "
               "the training frames runs in every view; foot points are projected to the ground and fused by "
               "visibility-normalised voting; people are tracked on the ground plane with anonymous, ephemeral ids; "
               "and the system reports only aggregates: zone counts, line crossings, dwell times and occupancy. "
               "Faces are blurred before anything is stored or drawn.",
               "`cffm-net-pilot` and `mrriandmstique/wildtrack-multi-camera-tracking`",
               "about 2 hours on 2 x T4",
               "`results/phase4/`: detection, tracking and count tables, the operator board and a clip"),
        SETUP,
        code('''
from pathlib import Path
import json
import cv2, numpy as np, pandas as pd, torch
import matplotlib.pyplot as plt
from cffm import multicam as mc, hud, showcase as sc
from ultralytics import YOLO

INPUT = Path("/kaggle/input") if ON_KAGGLE else env.project / "data" / "raw"
ROOT = next(p.parent for p in INPUT.rglob("calibrations") if (p.parent / "annotations_positions").exists())
views = mc.load_views(ROOT)
out = env.project / "results" / "phase4"
out.mkdir(parents=True, exist_ok=True)
val = [f for f in mc.FRAMES if mc.TEST_FROM - 200 <= f < mc.TEST_FROM]
test = [f for f in mc.FRAMES if f >= mc.TEST_FROM]
print("WILDTRACK:", ROOT, "| validation frames:", len(val), "| test frames:", len(test))
'''),
        md("## 1. Geometry check\n\nEvery annotated person projected through the calibration must land on the feet "
           "of their box in every view that sees them."),
        code('''
errs = []
for p in mc.load_annotations(ROOT, 0):
    xy = mc.position_to_cm([p["positionID"]])
    for v in p["views"]:
        if v["xmin"] >= 0:
            uv = views[v["viewNum"]].to_image(xy)[0]
            errs.append(np.linalg.norm(uv - [(v["xmin"] + v["xmax"]) / 2, v["ymax"]]) / (v["ymax"] - v["ymin"]))
print(f"median foot error {np.median(errs) * 100:.1f}% of person height over {len(errs)} boxes")
'''),
        md("## 2. Fine-tune the per-view detector\n\nYOLO26-m, COCO-pretrained, fine-tuned on the per-view boxes of "
           "the first 320 frames. Frames 1600 to 1795 validate, and the last 40 frames, the standard 10 % test "
           "split, are never seen in training or tuning."),
        code('''
ds = mc.export_yolo(ROOT, env.data / "wildtrack")
print("exported", ds)
run = env.runs / "wildtrack_yolo26m"
best, last = run / "weights" / "best.pt", run / "weights" / "last.pt"
if not best.exists() or last.exists() and not (run / "done").exists():
    m = YOLO(str(last) if last.exists() else "yolo26m.pt")
    m.train(data=str(env.data / "wildtrack" / "data.yaml"), imgsz=1280, epochs=40, batch=8, device=env.device,
            project=str(env.runs), name="wildtrack_yolo26m", exist_ok=True, close_mosaic=5, resume=last.exists(),
            plots=False, workers=4)
    (run / "done").touch()
det_models = {"zero-shot YOLO26-m": YOLO("yolo26m.pt"), "fine-tuned YOLO26-m": YOLO(str(best))}
'''),
        md("## 3. Detect in every view"),
        code('''
cache = out / "detections.npz"
if cache.exists():
    D = {k: v.item() for k, v in np.load(cache, allow_pickle=True).items()}
else:
    D = {}
    for name, m in det_models.items():
        D[name] = {}
        for f in val + test:
            ims = [str(mc.image_path(ROOT, c, f)) for c in range(len(views))]
            res = m.predict(ims, imgsz=1280, conf=0.05, classes=[0], verbose=False, half=True)
            D[name][f] = [np.column_stack([r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(),
                                           r.boxes.cls.cpu().numpy()]) for r in res]
    np.savez(cache, **{k: np.array(v, dtype=object) for k, v in D.items()})
print({k: len(v) for k, v in D.items()})
'''),
        md("## 4. Ground-plane detection\n\nFusion parameters are chosen on the validation frames only, then fixed. "
           "MODA, MODP, precision and recall at the standard 0.5 m radius on the test frames."),
        code('''
def ground(name, f, **kw):
    per = [mc.foot_points(views[c], D[name][f][c]) for c in range(len(views))]
    return mc.vote(views, [p for p, _ in per], [s for _, s in per], **kw)[0]


grid = [dict(radius=r, min_ratio=q, min_score=s) for r in (40, 60, 80) for q in (0.3, 0.5) for s in (0.15, 0.3, 0.45)]
gt = {f: mc.gt_positions(ROOT, f) for f in val + test}
res, chosen = {}, {}
for name in D:
    chosen[name] = max(grid, key=lambda kw: mc.moda([ground(name, f, **kw) for f in val], [gt[f] for f in val])["MODA"])
    res[name] = mc.moda([ground(name, f, **chosen[name]) for f in test], [gt[f] for f in test])
(pd.DataFrame(res).T[["MODA", "MODP", "precision", "recall"]] * 100).round(1)
'''),
        md("## 5. Anonymous tracking and counts\n\nTracks live on the ground plane only, with ids that exist for the "
           "life of a track and are never linked to appearance. The exit test of Phase 4 is counts that agree with "
           "a manual count; here the annotated positions are that count."),
        code('''
name = "fine-tuned YOLO26-m"
frames = val + test
trk = mc.GroundTracker(gate=80, max_miss=3, min_hits=2)
track_frames, pred = {}, {}
for f in frames:
    pred[f] = ground(name, f, **chosen[name])
    track_frames[f] = [(t.tid, *t.xy) for t in trk.update(pred[f], f // 5)]
zones = mc.default_zones()
mot = mc.ground_mot({f: track_frames[f] for f in test}, ROOT, test)
counts = mc.count_agreement([pred[f] for f in test], [gt[f] for f in test], zones)
tracks = trk.all_tracks()
ana = {"tracking": mot, "counts": counts, "crossings of y = 0": mc.line_crossings(tracks, 0.0),
       "dwell": mc.dwell_seconds(tracks, zones, fps=2)}
pd.DataFrame({"MOTA": [mot["mota"]], "IDF1": [mot["idf1"]], "ID switches": [mot["num_switches"]],
              "count MAE": [counts["count_mae"]], "mean people": [counts["mean_people"]],
              "relative count error": [counts["count_rel_err"]]}).round(3)
'''),
        md("## 6. The operator board\n\nSeven views with faces blurred at ingest, beside the ground plane seen from "
           "above: every anonymous track with its trail, the zones and their live counts."),
        code('''
SCALE = 0.35


def to_px(xy):
    return ((np.asarray(xy, float) - mc.ORIGIN_CM) * SCALE).astype(int)


def bev(f):
    W, H = int(mc.GRID[0] * mc.CELL_CM * SCALE), int(mc.GRID[1] * mc.CELL_CM * SCALE)
    img = np.full((H, W, 3), (20, 16, 14), np.uint8)
    for z in zones:
        (x0, y0), (x1, y1) = to_px([z.x[0], z.y[0]]), to_px([z.x[1], z.y[1]])
        cv2.rectangle(img, (int(x0), int(y0)), (int(x1) - 1, int(y1) - 1), (76, 66, 60), 1)
        cv2.putText(img, f"{z.name.upper()}  {int(z.contains(pred[f]).sum())}", (int(x0) + 8, int(y0) + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1, cv2.LINE_AA)
    for t in tracks:
        path = [p for p in t.path if p[0] <= f // 5][-12:]
        if path and path[-1][0] == f // 5:
            pts = to_px([p[1:] for p in path])
            cv2.polylines(img, [pts.reshape(-1, 1, 2)], False, (0, 160, 255), 2, cv2.LINE_AA)
            cv2.circle(img, (int(pts[-1][0]), int(pts[-1][1])), 7, (0, 204, 255), 2, cv2.LINE_AA)
            cv2.putText(img, f"{t.tid:03d}", (int(pts[-1][0]) + 9, int(pts[-1][1]) - 9), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, (0, 204, 255), 1, cv2.LINE_AA)
    return img


def board(f):
    tiles = []
    for c in range(len(views)):
        im = cv2.imread(str(mc.image_path(ROOT, c, f)))
        d = D[name][f][c]
        d = d[d[:, 4] >= 0.3]
        im = mc.anonymize(im, d[:, :4])
        for x1, y1, x2, y2, *_ in d:
            cv2.rectangle(im, (int(x1), int(y1)), (int(x2), int(y2)), (0, 204, 255), 3)
        tiles.append(cv2.resize(im, (640, 360)))
    left = np.concatenate([np.concatenate(tiles[:4], 1), np.concatenate(tiles[4:] + [np.zeros_like(tiles[0])], 1)], 0)
    b = bev(f)
    right = cv2.resize(b, (int(left.shape[0] * b.shape[1] / b.shape[0]), left.shape[0]))
    return np.concatenate([left, right], 1)[..., ::-1]


boards = [board(f) for f in test]
hud.save_video(boards, out / "multicamera.mp4", fps=2, width=1920)
sc.show(boards[len(boards) // 2], width=1400)
'''),
        code('''
hm = mc.heatmap(np.concatenate([pred[f] for f in frames if len(pred[f])]))
plt.figure(figsize=(3, 7))
plt.imshow(hm, cmap="inferno")
plt.axis("off")
plt.title("occupancy")
plt.savefig(out / "heatmap.png", dpi=150, bbox_inches="tight")
(out / "phase4_summary.json").write_text(json.dumps({"detection": res, "fusion_params": chosen, **ana},
                                                    indent=2, default=float))
print("saved", sorted(p.name for p in out.iterdir()))
'''),
        CLEANUP,
    ])
