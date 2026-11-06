"""Phase 3: live paired cameras with capture threads, newest-frame queues, a stream monitor and causal inference."""
from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field

import cv2
import numpy as np
import torch


@dataclass
class Frame:
    image: np.ndarray
    ts: float
    seq: int
    source_index: int = -1


class Latest:
    """A one-slot queue: put replaces the waiting frame, so a slow consumer never sees a backlog."""

    def __init__(self):
        self._f, self._cv, self.dropped, self.put_count = None, threading.Condition(), 0, 0

    def put(self, f: Frame):
        with self._cv:
            if self._f is not None:
                self.dropped += 1
            self._f = f
            self.put_count += 1
            self._cv.notify_all()

    def peek(self) -> Frame | None:
        with self._cv:
            return self._f

    def take(self, timeout=None) -> Frame | None:
        with self._cv:
            if self._f is None:
                self._cv.wait(timeout)
            f, self._f = self._f, None
            return f


class Camera(threading.Thread):
    """Capture thread: read from `open_fn()` forever, timestamp at capture, reconnect with backoff on failure."""

    def __init__(self, name, open_fn, grey=False, reconnect=0.5):
        super().__init__(daemon=True, name=f"cam-{name}")
        self.cam_name, self.open_fn, self.grey, self.reconnect = name, open_fn, grey, reconnect
        self.q, self.stop_ev, self.n, self.reconnects = Latest(), threading.Event(), 0, 0

    def run(self):
        while not self.stop_ev.is_set():
            src = self.open_fn()
            if src is None:
                time.sleep(self.reconnect)
                continue
            while not self.stop_ev.is_set():
                ok, img, idx = src.read()
                if not ok:
                    break
                ts = time.perf_counter()
                if self.grey and img.ndim == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                self.q.put(Frame(img, ts, self.n, idx))
                self.n += 1
            src.release()
            if not self.stop_ev.is_set():
                self.reconnects += 1
                time.sleep(self.reconnect)

    def stop(self):
        self.stop_ev.set()


class DeviceSource:
    """A webcam index or RTSP/HTTP URL through OpenCV; RTSP over TCP."""

    def __init__(self, url):
        if isinstance(url, str) and url.startswith("rtsp"):
            import os
            os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")
        self.cap = cv2.VideoCapture(url)

    def read(self):
        ok, img = self.cap.read()
        return ok, img, -1

    def release(self):
        self.cap.release()


@dataclass
class StreamFaults:
    """Faults of a live link, keyed by wall-clock seconds since start: stalls, freezes, unplugs, drops, delay."""
    stalls: list = field(default_factory=list)
    freezes: list = field(default_factory=list)
    unplugs: list = field(default_factory=list)
    drop_p: float = 0.0
    delay_s: float = 0.0
    jitter_s: float = 0.0

    @staticmethod
    def _in(windows, t):
        return any(a <= t < b for a, b in windows)


class ReplaySource:
    """Plays an image sequence in real time at `fps`, as a camera would, with optional link faults."""

    def __init__(self, paths, fps=30.0, grey=False, faults: StreamFaults | None = None, t0=None, loop=True, seed=0):
        self.paths, self.fps, self.grey, self.faults = list(paths), fps, grey, faults or StreamFaults()
        self.t0 = t0 if t0 is not None else time.perf_counter()
        self.loop, self.rng, self.last = loop, np.random.default_rng(seed), None
        self.k = max(0, int(np.ceil((time.perf_counter() - self.t0) * fps)))

    def read(self):
        f = self.faults
        while True:
            due = self.t0 + self.k / self.fps + f.delay_s + (self.rng.uniform(0, f.jitter_s) if f.jitter_s else 0)
            now = time.perf_counter()
            if due > now:
                time.sleep(due - now)
            t = due - self.t0
            if StreamFaults._in(f.unplugs, t):
                self.k += 1
                return False, None, -1
            i = self.k % len(self.paths) if self.loop else self.k
            self.k += 1
            if i >= len(self.paths):
                return False, None, -1
            if StreamFaults._in(f.stalls, t) or (f.drop_p and self.rng.random() < f.drop_p):
                continue
            if StreamFaults._in(f.freezes, t) and self.last is not None:
                return True, self.last[0].copy(), self.last[1]
            img = cv2.imread(str(self.paths[i]), cv2.IMREAD_GRAYSCALE if self.grey else cv2.IMREAD_COLOR)
            self.last = (img, i)
            return True, img, i

    def release(self):
        pass


@dataclass
class Health:
    status: str = "ok"
    fps: float = 0.0
    age_ms: float = 0.0
    frozen: bool = False
    reasons: tuple = ()


class Monitor:
    """Per-camera health from arrival times and pixels: stalled, frozen, black, saturated; raises sensor flags."""

    def __init__(self, nominal_fps=30.0, stall_intervals=4.0, freeze_frames=6, dark=8.0, glare=0.35):
        self.nominal, self.stall_s = nominal_fps, stall_intervals / nominal_fps
        self.freeze_frames, self.dark, self.glare = freeze_frames, dark, glare
        self.arrivals, self.sig, self.same = {}, {}, {}

    def observe(self, name, f: Frame | None, now):
        a = self.arrivals.setdefault(name, deque(maxlen=30))
        if f is not None and (not a or f.ts != a[-1]):
            a.append(f.ts)
            small = cv2.resize(f.image, (32, 24), interpolation=cv2.INTER_AREA).astype(np.int16)
            prev = self.sig.get(name)
            self.same[name] = self.same.get(name, 0) + 1 if prev is not None and np.abs(small - prev).mean() < 0.5 else 0
            self.sig[name] = small

    def health(self, name, f: Frame | None, now) -> Health:
        a = self.arrivals.get(name, ())
        fps = (len(a) - 1) / (a[-1] - a[0]) if len(a) > 1 and a[-1] > a[0] else 0.0
        age = (now - a[-1]) if a else float("inf")
        reasons = []
        if age > self.stall_s:
            reasons.append("stalled")
        frozen = self.same.get(name, 0) >= self.freeze_frames
        if frozen:
            reasons.append("frozen")
        if f is not None:
            m = float(f.image.mean())
            if m < self.dark:
                reasons.append("black")
            sat = float((f.image >= 250).mean())
            if sat > self.glare:
                reasons.append("saturated")
        down = bool({"stalled", "frozen", "black"} & set(reasons))
        status = "down" if down else "degraded" if reasons else "ok"
        return Health(status, fps, age * 1000 if np.isfinite(age) else float("inf"), frozen, tuple(reasons))


class Runtime:
    """Main loop: newest pair, health, flags, causal model step with real time gaps, tracking, records."""

    def __init__(self, model, visible: Camera, thermal: Camera, *, fps=30.0, imgsz=(512, 640), device="cuda",
                 memory=True, use_monitor=True, track=True, conf=0.25, reset_after_s=2.0, render=None, tau0=0.1):
        self.model, self.vis, self.ir, self.fps = model.eval(), visible, thermal, fps
        self.h, self.w = imgsz
        self.device, self.memory, self.use_monitor, self.conf = device, memory, use_monitor, conf
        self.monitor, self.records, self.reset_after = Monitor(fps), [], reset_after_s
        self.render, self.tau0 = render, tau0
        if getattr(model, "memory", None) is not None:
            model.memory_on = memory
            model.reset_memory()
        if track:
            from .video import tracker
            self.tracker = tracker()
        else:
            self.tracker = None
        self._last_ts, self._last = None, {"visible": None, "thermal": None}

    def _tensor(self, vis, ir):
        v = vis.image if vis is not None else np.zeros((self.h, self.w, 3), np.uint8)
        t = ir.image if ir is not None else np.zeros((self.h, self.w), np.uint8)
        if v.shape[:2] != (self.h, self.w):
            v = cv2.resize(v, (self.w, self.h))
        if t.shape[:2] != (self.h, self.w):
            t = cv2.resize(t, (self.w, self.h))
        x = np.concatenate([v, t[..., None]], 2)
        return torch.from_numpy(x).to(self.device, non_blocking=True).permute(2, 0, 1)[None].float().div_(255)

    @torch.no_grad()
    def step(self):
        vis_new = self.vis.q.take(timeout=1.0 / self.fps)
        ir_new = self.ir.q.take(timeout=0)
        now = time.perf_counter()
        for k, f in (("visible", vis_new), ("thermal", ir_new)):
            self.monitor.observe(k, f, now)
            if f is not None:
                self._last[k] = f
        vis, ir = self._last["visible"], self._last["thermal"]
        if vis is None and ir is None:
            return None
        hv, ht = self.monitor.health("visible", vis, now), self.monitor.health("thermal", ir, now)
        flags = [0.0 if hv.status == "down" else 1.0, 0.0 if ht.status == "down" else 1.0]
        x = self._tensor(vis if flags[0] else None, ir if flags[1] else None)
        newest = max(f.ts for f in (vis, ir) if f is not None)
        gap = (newest - self._last_ts) if self._last_ts is not None else 1.0 / self.fps
        self._last_ts = newest
        m = self.model
        if getattr(m, "memory", None) is not None and self.memory:
            reset = gap > self.reset_after
            m.memory_context(tau_ratio=torch.tensor([max(gap / self.tau0, 0.0)], device=self.device),
                             reset=torch.tensor([reset], device=self.device))
        m.sensor_flags = torch.tensor([flags], device=self.device) if (self.use_monitor and min(flags) < 1) else None
        t0 = time.perf_counter()
        from .video import detect, _Dets
        with torch.autocast("cuda", torch.float16, enabled=x.is_cuda):
            d = detect(m, x, conf=self.conf)[0]
        if x.is_cuda:
            torch.cuda.synchronize()
        t_inf = time.perf_counter() - t0
        tracks = self.tracker.update(_Dets(d.reshape(-1, 6))) if self.tracker is not None else d
        done = time.perf_counter()
        rec = {"t": done, "infer_ms": t_inf * 1000, "e2e_ms": (done - newest) * 1000, "gap_ms": gap * 1000,
               "vis_idx": vis.source_index if vis else -1, "ir_idx": ir.source_index if ir else -1,
               "desync_ms": abs(vis.ts - ir.ts) * 1000 if vis and ir else float("nan"),
               "vis": hv.status, "ir": ht.status, "vis_why": hv.reasons, "ir_why": ht.reasons,
               "n_det": int(len(d)), "dets": d, "tracks": np.asarray(tracks),
               "gpu_mb": torch.cuda.memory_allocated() / 2**20 if x.is_cuda else 0.0}
        self.records.append(rec)
        if self.render is not None:
            self.render(vis, ir, rec)
        return rec

    def run(self, seconds):
        for cam in (self.vis, self.ir):
            if not cam.is_alive():
                cam.start()
        end = time.perf_counter() + seconds
        while time.perf_counter() < end:
            self.step()
        return self.records

    def close(self):
        self.vis.stop()
        self.ir.stop()
        self.model.sensor_flags = None


def summary(records, skip_s=2.0) -> dict:
    """Throughput, latency percentiles, drift (late minus early e2e) and memory growth over a run."""
    if not records:
        return {}
    t0 = records[0]["t"]
    r = [x for x in records if x["t"] - t0 >= skip_s] or records
    e2e = np.array([x["e2e_ms"] for x in r])
    inf = np.array([x["infer_ms"] for x in r])
    dur = r[-1]["t"] - r[0]["t"]
    q = max(len(r) // 5, 1)
    return {"seconds": dur, "fps": (len(r) - 1) / dur if dur > 0 else 0.0,
            "e2e_p50": float(np.median(e2e)), "e2e_p95": float(np.percentile(e2e, 95)),
            "infer_p50": float(np.median(inf)), "infer_p95": float(np.percentile(inf, 95)),
            "drift_ms": float(np.median(e2e[-q:]) - np.median(e2e[:q])),
            "gpu_growth_mb": float(r[-1]["gpu_mb"] - r[0]["gpu_mb"])}


def detection_delay(records, onset_s, offset_s, camera="ir", start=None) -> dict:
    """Seconds from a fault's onset to the first 'down', and from its end to the first 'ok' again."""
    t0 = start if start is not None else records[0]["t"]
    down = [x["t"] - t0 for x in records if x[camera] == "down" and x["t"] - t0 >= onset_s]
    up = [x["t"] - t0 for x in records if x[camera] == "ok" and x["t"] - t0 >= offset_s]
    false = [x for x in records if x[camera] == "down" and not (onset_s <= x["t"] - t0 < offset_s + 1.0)]
    return {"detect_s": (down[0] - onset_s) if down else float("inf"),
            "recover_s": (up[0] - offset_s) if up else float("inf"),
            "false_alarm_rate": len(false) / max(len(records), 1)}
