const DEFAULT_BACKEND = "";
const GRADIO_CLIENT = "https://cdn.jsdelivr.net/npm/@gradio/client/dist/index.min.js";
const SAMPLES = [
  ["llvip_190311", "Night pedestrians · LLVIP"], ["llvip_190009", "Night pedestrians · LLVIP"],
  ["llvip_190505", "Night pedestrians · LLVIP"], ["llvip_190648", "Night pedestrians · LLVIP"],
  ["m3fd_01967", "City traffic · M3FD"], ["m3fd_03398", "City traffic · M3FD"], ["m3fd_03401", "City traffic · M3FD"],
];
const PHOTOS = ["photo_bus", "photo_zidane"];
const EPHEMERAL = [".gradio.live", ".trycloudflare.com"];

const $ = (id) => document.getElementById(id);
const store = { get: (k) => { try { return localStorage.getItem(k); } catch { return null; } },
                set: (k, v) => { try { localStorage.setItem(k, v); } catch {} } };
const state = { mode: "pair", vis: null, ir: null, client: null, url: null, sample: null };

const saved = store.get("cffm-backend");
$("backend").value = saved && !EPHEMERAL.some((h) => saved.includes(h)) ? saved : DEFAULT_BACKEND;
$("backend").addEventListener("change", (e) => { store.set("cffm-backend", e.target.value.trim()); state.client = null; });
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
  if (!df || !df.headers || !df.data || !df.data.length) return "";
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
  status("loading the recorded result …");
  const rec = await (await fetch(`assets/boards/${name}.json`)).json();
  show(`assets/boards/${name}.jpg`, rec.log, `${rec.msg} · recorded run at confidence 0.30`);
  status("recorded result shown");
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
  try { await showSample(it.s); } catch (err) { status("could not load the recorded result", true); }
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

async function client() {
  const url = $("backend").value.trim();
  if (!url) throw new Error("no live backend is set");
  if (!state.client || state.url !== url) {
    status("connecting to the live model …");
    const { Client } = await import(GRADIO_CLIENT);
    state.client = await Client.connect(url);
    state.url = url;
  }
  return state.client;
}

async function analyze() {
  const pair = state.mode === "pair";
  if (!state.vis || (pair && !state.ir)) { status(pair ? "Add both images first, or pick a sample." : "Add a photo first, or pick a sample.", true); return; }
  const conf = Number($("conf").value);
  if (state.sample && conf === 0.3) { showSample(state.sample); return; }
  $("go").disabled = true;
  try {
    const app = await client();
    const { handle_file } = await import(GRADIO_CLIENT);
    status("analyzing …");
    const res = pair ? await app.predict("/analyze_pair", [handle_file(state.vis), handle_file(state.ir), $("model").value, conf])
                     : await app.predict("/analyze_photo", [handle_file(state.vis), conf]);
    const [img, df, msg] = res.data;
    show(img.url || img, df, msg);
    status(msg || "done");
  } catch (err) {
    console.error(err);
    status("The live model is offline right now, so uploads cannot be analyzed. The samples below show recorded results from the same model.", true);
  } finally {
    $("go").disabled = false;
  }
}

function paintTheme() {
  const light = document.documentElement.dataset.theme === "light";
  $("theme").textContent = light ? "Dark" : "Light";
  $("theme").setAttribute("aria-label", light ? "Switch to the dark theme" : "Switch to the light theme");
}
$("theme").addEventListener("click", () => {
  const light = document.documentElement.dataset.theme === "light";
  if (light) delete document.documentElement.dataset.theme; else document.documentElement.dataset.theme = "light";
  store.set("cffm-theme", light ? "dark" : "light");
  paintTheme();
});
paintTheme();

$("go").addEventListener("click", analyze);
renderSamples();
