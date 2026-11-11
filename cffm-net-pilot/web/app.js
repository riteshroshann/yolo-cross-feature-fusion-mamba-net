const ORT = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/";
const SAMPLES = [
  ["llvip_190311", "llvip_cffm"], ["llvip_190009", "llvip_cffm"], ["llvip_190505", "llvip_cffm"], ["llvip_190648", "llvip_cffm"],
  ["m3fd_01967", "m3fd_cffm"], ["m3fd_03398", "m3fd_cffm"], ["m3fd_03401", "m3fd_cffm"],
];
const PHOTOS = ["photo_bus", "photo_zidane"];
const SIZE = 640;
const PAD = 114 / 255;
const IOU = 0.7;
const PALETTE = ["#ffcc00", "#00d4ff", "#ff7c28", "#7dff8a", "#ff5d9e", "#b48cff"];

const $ = (id) => document.getElementById(id);
const store = { get: (k) => { try { return localStorage.getItem(k); } catch { return null; } },
                set: (k, v) => { try { localStorage.setItem(k, v); } catch {} } };
const state = { mode: "pair", vis: null, ir: null, sample: null, ort: null, meta: null, sessions: {} };

$("conf").addEventListener("input", (e) => { $("confOut").value = Number(e.target.value).toFixed(2); });

function status(text, err = false) { $("status").textContent = text; $("status").classList.toggle("err", err); }

function setFile(which, blob) {
  state[which] = blob;
  const drop = $(which === "vis" ? "dropVis" : "dropIr");
  drop.querySelector("img").src = URL.createObjectURL(blob);
  drop.classList.add("has");
}

function userFile(which, blob) {
  state.sample = null;
  document.querySelectorAll(".samples button").forEach((b) => b.classList.remove("on"));
  setFile(which, blob);
}

for (const [id, which] of [["dropVis", "vis"], ["dropIr", "ir"]]) {
  const el = $(id);
  el.querySelector("input").addEventListener("change", (e) => e.target.files[0] && userFile(which, e.target.files[0]));
  el.addEventListener("dragover", (e) => { e.preventDefault(); el.classList.add("over"); });
  el.addEventListener("dragleave", () => el.classList.remove("over"));
  el.addEventListener("drop", (e) => {
    e.preventDefault(); el.classList.remove("over");
    const files = [...e.dataTransfer.files].filter((f) => f.type.startsWith("image/"));
    if (files.length === 2 && state.mode === "pair") { userFile("vis", files[0]); userFile("ir", files[1]); }
    else if (files[0]) userFile(which, files[0]);
  });
}

function table(df) {
  if (!df || !df.data.length) return "";
  const head = df.headers.map((h) => `<th>${h}</th>`).join("");
  const rows = df.data.map((r) => `<tr>${r.map((c) => `<td>${c ?? ""}</td>`).join("")}</tr>`).join("");
  return `<div class="tablewrap"><table><thead><tr>${head}</tr></thead><tbody>${rows}</tbody></table></div>`;
}

function show(img, df, msg) {
  $("board").src = img;
  $("result").hidden = false;
  $("caption").textContent = msg || "";
  $("log").innerHTML = table(df);
}

async function showSample(name) {
  status("Loading the recorded result …");
  const rec = await (await fetch(`assets/boards/${name}.json`)).json();
  show(`assets/boards/${name}.jpg`, rec.log, `${rec.msg} · recorded run at confidence 0.30`);
  status("Recorded result shown. Move the confidence slider and analyze to run the model in your browser.");
}

async function pickSample(it, button) {
  const get = async (u) => (await fetch(u)).blob();
  document.querySelectorAll(".samples button").forEach((b) => b.classList.toggle("on", b === button));
  if (state.mode === "pair") {
    setFile("vis", await get(`assets/samples/${it.s}_visible.jpg`));
    setFile("ir", await get(`assets/samples/${it.s}_thermal.jpg`));
    $("model").value = it.m;
  } else {
    setFile("vis", await get(it.thumb));
  }
  state.sample = it.s;
  $("conf").value = 0.3;
  $("confOut").value = "0.30";
  try { await showSample(it.s); } catch { status("Could not load the recorded result. Check your connection and pick it again.", true); }
}

function renderSamples() {
  const box = $("samples");
  box.innerHTML = "";
  const items = state.mode === "pair" ? SAMPLES.map(([s, m]) => ({ s, m, thumb: `assets/samples/${s}_visible.jpg` }))
                                      : PHOTOS.map((s) => ({ s, thumb: `assets/samples/${s}.jpg` }));
  for (const it of items) {
    const b = document.createElement("button");
    b.type = "button";
    const label = it.s.replace("photo_", "").replace("_", " ");
    b.innerHTML = `<img src="${it.thumb}" alt="Sample ${label}" width="168" height="108" loading="lazy"><small>${label}</small>`;
    b.addEventListener("click", () => pickSample(it, b));
    box.appendChild(b);
  }
}

document.querySelectorAll(".tab").forEach((t) => t.addEventListener("click", () => {
  document.querySelectorAll(".tab").forEach((x) => { x.classList.toggle("on", x === t); x.setAttribute("aria-selected", x === t); });
  state.mode = t.dataset.mode;
  state.sample = null;
  document.querySelector(".panel").classList.toggle("photo", state.mode === "photo");
  renderSamples();
}));

async function session(name) {
  if (!state.ort) {
    status("Loading the runtime …");
    state.ort = await import(ORT + "ort.wasm.min.mjs");
    state.ort.env.wasm.wasmPaths = ORT;
    state.ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 1) : 1;
    state.meta = await (await fetch("models/meta.json")).json();
  }
  if (!state.sessions[name]) {
    status(`Downloading the model (${state.meta[name].mb} MB, first time only) …`);
    state.sessions[name] = await state.ort.InferenceSession.create(`models/${name}.onnx`, { executionProviders: ["wasm"] });
  }
  return state.sessions[name];
}

async function bitmap(blob) {
  try { return await createImageBitmap(blob); } catch { throw new Error("that file is not an image the browser can read"); }
}

function letterbox(img) {
  const r = Math.min(SIZE / img.width, SIZE / img.height, 1);
  const w = Math.round(img.width * r), h = Math.round(img.height * r);
  return { r, w, h, dx: Math.floor((SIZE - w) / 2), dy: Math.floor((SIZE - h) / 2) };
}

function pixels(img, box) {
  const c = new OffscreenCanvas(SIZE, SIZE);
  const g = c.getContext("2d", { willReadFrequently: true });
  g.drawImage(img, box.dx, box.dy, box.w, box.h);
  return g.getImageData(0, 0, SIZE, SIZE).data;
}

function inside(box, i) {
  const x = i % SIZE, y = (i / SIZE) | 0;
  return x >= box.dx && x < box.dx + box.w && y >= box.dy && y < box.dy + box.h;
}

function tensor(vis, ir, box) {
  const n = SIZE * SIZE, ch = ir ? 4 : 3;
  const x = new Float32Array(ch * n).fill(PAD);
  const a = pixels(vis, box), b = ir ? pixels(ir, box) : null;
  for (let i = 0; i < n; i++) {
    if (!inside(box, i)) continue;
    const r = a[4 * i] / 255, g = a[4 * i + 1] / 255, bl = a[4 * i + 2] / 255;
    if (ir) {
      x[i] = bl; x[n + i] = g; x[2 * n + i] = r;
      x[3 * n + i] = (0.299 * b[4 * i] + 0.587 * b[4 * i + 1] + 0.114 * b[4 * i + 2]) / 255;
    } else {
      x[i] = r; x[n + i] = g; x[2 * n + i] = bl;
    }
  }
  return new state.ort.Tensor("float32", x, [1, ch, SIZE, SIZE]);
}

function iou(a, b) {
  const w = Math.min(a[2], b[2]) - Math.max(a[0], b[0]), h = Math.min(a[3], b[3]) - Math.max(a[1], b[1]);
  if (w <= 0 || h <= 0) return 0;
  const inter = w * h;
  return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter);
}

function decode(out, conf, box, img) {
  const [, rows, n] = out.dims, d = out.data, nc = rows - 4, found = [];
  for (let i = 0; i < n; i++) {
    let best = 0, cls = 0;
    for (let c = 0; c < nc; c++) { const s = d[(4 + c) * n + i]; if (s > best) { best = s; cls = c; } }
    if (best < conf) continue;
    const cx = d[i], cy = d[n + i], w = d[2 * n + i], h = d[3 * n + i];
    const px = (v, off, max) => Math.min(max, Math.max(0, (v - off) / box.r));
    found.push([px(cx - w / 2, box.dx, img.width), px(cy - h / 2, box.dy, img.height),
                px(cx + w / 2, box.dx, img.width), px(cy + h / 2, box.dy, img.height), best, cls]);
  }
  found.sort((p, q) => q[4] - p[4]);
  const kept = [];
  for (const f of found) if (!kept.some((k) => k[5] === f[5] && iou(k, f) > IOU)) kept.push(f);
  return kept.slice(0, 300);
}

function drawPanel(g, img, x, y, w, h, label, det, names, grey) {
  g.save();
  if (grey) g.filter = "grayscale(1)";
  g.drawImage(img, x, y, w, h);
  g.restore();
  const s = w / img.width, u = Math.max(2, w / 420);
  g.font = `600 ${Math.round(5.5 * u)}px ui-monospace, Consolas, monospace`;
  g.textBaseline = "top";
  det.forEach((b, i) => {
    const col = PALETTE[b[5] % PALETTE.length];
    const x1 = x + b[0] * s, y1 = y + b[1] * s, x2 = x + b[2] * s, y2 = y + b[3] * s, k = Math.min(6 * u, (x2 - x1) / 3, (y2 - y1) / 3);
    g.strokeStyle = col; g.lineWidth = u;
    g.beginPath();
    for (const [cx, cy, sx, sy] of [[x1, y1, 1, 1], [x2, y1, -1, 1], [x1, y2, 1, -1], [x2, y2, -1, -1]]) {
      g.moveTo(cx + sx * k, cy); g.lineTo(cx, cy); g.lineTo(cx, cy + sy * k);
    }
    g.stroke();
    const tag = `${String(i + 1).padStart(2, "0")} ${String(names[b[5]]).toUpperCase()} ${b[4].toFixed(2)}`;
    const tw = g.measureText(tag).width + 3 * u, th = 7.5 * u, ty = y1 - th - u < y ? y1 + u : y1 - th - u;
    g.fillStyle = col; g.fillRect(x1, ty, tw, th);
    g.fillStyle = "#08090c"; g.fillText(tag, x1 + 1.5 * u, ty + u);
  });
  g.fillStyle = "rgba(8, 9, 12, .78)"; g.fillRect(x + 4 * u, y + 4 * u, g.measureText(label).width + 6 * u, 9 * u);
  g.fillStyle = "#eceff2"; g.fillText(label, x + 7 * u, y + 6 * u);
}

function board(vis, ir, det, names, title, summary) {
  const pw = Math.min(1280, Math.max(640, vis.width)), ph = Math.round(pw * vis.height / vis.width);
  const gap = 10, head = Math.round(pw / 22), cols = ir ? 2 : 1;
  const c = document.createElement("canvas");
  c.width = pw * cols + gap * (cols + 1); c.height = head + ph + gap;
  const g = c.getContext("2d");
  g.fillStyle = "#08090c"; g.fillRect(0, 0, c.width, c.height);
  g.fillStyle = "#ffcc00"; g.fillRect(gap, head * 0.25, head * 0.12, head * 0.5);
  g.textBaseline = "middle";
  g.font = `700 ${Math.round(head * 0.4)}px ui-monospace, Consolas, monospace`;
  g.fillStyle = "#eceff2"; g.fillText(title, gap + head * 0.3, head / 2);
  g.font = `500 ${Math.round(head * 0.34)}px ui-monospace, Consolas, monospace`;
  g.fillStyle = "#8a94a3"; g.fillText(summary, c.width - gap - g.measureText(summary).width, head / 2);
  drawPanel(g, vis, gap, head, pw, ph, "VISIBLE", det, names, false);
  if (ir) drawPanel(g, ir, pw + 2 * gap, head, pw, ph, "THERMAL", det, names, true);
  return c.toDataURL("image/jpeg", 0.9);
}

function summarize(det, names, ms) {
  const counts = {};
  for (const b of det) counts[names[b[5]]] = (counts[names[b[5]]] || 0) + 1;
  const parts = Object.entries(counts).sort((p, q) => q[1] - p[1]).map(([n, c]) => `${String(n).toUpperCase()} x${c}`);
  return (parts.length ? parts.join("  ") : "NO SUBJECTS") + `  ${Math.round(ms)} MS`;
}

async function runInBrowser() {
  const pair = state.mode === "pair";
  const name = pair ? $("model").value : "yolo26n";
  const sess = await session(name);
  const vis = await bitmap(state.vis), ir = pair ? await bitmap(state.ir) : null;
  const box = letterbox(vis);
  status("Analyzing in your browser …");
  await new Promise((r) => setTimeout(r, 30));
  const t = performance.now();
  const out = await sess.run({ images: tensor(vis, ir, box) });
  const ms = performance.now() - t;
  const names = state.meta[name].names;
  const det = decode(out.detections, Number($("conf").value), box, vis);
  const title = pair ? `CFFM-NET // ${name.split("_")[0].toUpperCase()}` : "YOLO26-N // COCO, 80 CLASSES (NOT CFFM-NET)";
  const rows = det.map((b, i) => [String(i + 1).padStart(2, "0"), String(names[b[5]]).toUpperCase(), b[4].toFixed(3),
                                  `${Math.round(b[2] - b[0])}x${Math.round(b[3] - b[1])}`]);
  const msg = `${det.length} subjects · ${Math.round(ms)} ms in your browser`;
  show(board(vis, ir, det, names, title, summarize(det, names, ms)), { headers: ["subject", "class", "conf", "box px"], data: rows }, msg);
  status(msg);
}

async function analyze() {
  const pair = state.mode === "pair";
  if (!state.vis || (pair && !state.ir)) { status(pair ? "Add both images first, or pick a sample." : "Add a photo first, or pick a sample.", true); return; }
  if (state.sample && Number($("conf").value) === 0.3) { showSample(state.sample); return; }
  $("go").disabled = true;
  try {
    await runInBrowser();
  } catch (err) {
    console.error(err);
    status(`The model could not run in this browser: ${err.message || err}. The samples still show recorded results.`, true);
  } finally {
    $("go").disabled = false;
  }
}

function paintTheme() {
  const dark = document.documentElement.dataset.theme === "dark";
  $("theme").setAttribute("aria-label", dark ? "Switch to the light theme" : "Switch to the dark theme");
  $("theme").setAttribute("aria-pressed", dark);
}

$("theme").addEventListener("click", () => {
  const dark = document.documentElement.dataset.theme === "dark";
  if (dark) delete document.documentElement.dataset.theme; else document.documentElement.dataset.theme = "dark";
  store.set("cffm-theme", dark ? "light" : "dark");
  paintTheme();
});
paintTheme();

$("go").addEventListener("click", analyze);
renderSamples();
