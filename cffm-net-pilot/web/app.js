import { Client, handle_file } from "https://cdn.jsdelivr.net/npm/@gradio/client/dist/index.min.js";

const DEFAULT_BACKEND = "https://bbac94ca05251fde7e.gradio.live/";
const SAMPLES = [
  ["llvip_190311", "Night pedestrians · LLVIP"], ["llvip_190009", "Night pedestrians · LLVIP"],
  ["llvip_190505", "Night pedestrians · LLVIP"], ["llvip_190648", "Night pedestrians · LLVIP"],
  ["m3fd_01967", "City traffic · M3FD"], ["m3fd_03398", "City traffic · M3FD"], ["m3fd_03401", "City traffic · M3FD"],
];
const PHOTOS = ["photo_bus", "photo_zidane"];

const $ = (id) => document.getElementById(id);
const store = { get: (k) => { try { return localStorage.getItem(k); } catch { return null; } },
                set: (k, v) => { try { localStorage.setItem(k, v); } catch {} } };
const state = { mode: "pair", vis: null, ir: null, client: null, url: null };

$("backend").value = store.get("cffm-backend") || DEFAULT_BACKEND;
$("backend").addEventListener("change", (e) => { store.set("cffm-backend", e.target.value.trim()); state.client = null; });
$("conf").addEventListener("input", (e) => { $("confOut").value = Number(e.target.value).toFixed(2); });

function status(text, err = false) { $("status").textContent = text; $("status").classList.toggle("err", err); }

function setFile(which, blob) {
  state[which] = blob;
  const drop = $(which === "vis" ? "dropVis" : "dropIr");
  drop.querySelector("img").src = URL.createObjectURL(blob);
  drop.classList.add("has");
}

for (const [id, which] of [["dropVis", "vis"], ["dropIr", "ir"]]) {
  const el = $(id);
  el.querySelector("input").addEventListener("change", (e) => e.target.files[0] && setFile(which, e.target.files[0]));
  el.addEventListener("dragover", (e) => { e.preventDefault(); el.classList.add("over"); });
  el.addEventListener("dragleave", () => el.classList.remove("over"));
  el.addEventListener("drop", (e) => {
    e.preventDefault(); el.classList.remove("over");
    const files = [...e.dataTransfer.files].filter((f) => f.type.startsWith("image/"));
    if (files.length === 2 && state.mode === "pair") { setFile("vis", files[0]); setFile("ir", files[1]); }
    else if (files[0]) setFile(which, files[0]);
  });
}

function renderSamples() {
  const box = $("samples");
  box.innerHTML = "";
  const items = state.mode === "pair" ? SAMPLES.map(([s, m]) => ({ s, m, thumb: `assets/samples/${s}_visible.jpg` }))
                                      : PHOTOS.map((s) => ({ s, thumb: `assets/samples/${s}.jpg` }));
  for (const it of items) {
    const b = document.createElement("button");
    b.innerHTML = `<img src="${it.thumb}" alt="${it.s}"><small>${it.s.replace("photo_", "")}</small>`;
    b.addEventListener("click", async () => {
      const get = async (u) => (await fetch(u)).blob();
      if (state.mode === "pair") {
        setFile("vis", await get(`assets/samples/${it.s}_visible.jpg`));
        setFile("ir", await get(`assets/samples/${it.s}_thermal.jpg`));
        $("model").value = it.m;
      } else {
        setFile("vis", await get(it.thumb));
      }
      analyze();
    });
    box.appendChild(b);
  }
}

document.querySelectorAll(".tab").forEach((t) => t.addEventListener("click", () => {
  document.querySelectorAll(".tab").forEach((x) => x.classList.toggle("on", x === t));
  state.mode = t.dataset.mode;
  document.querySelector(".panel").classList.toggle("photo", state.mode === "photo");
  renderSamples();
}));

async function client() {
  const url = $("backend").value.trim();
  if (!state.client || state.url !== url) {
    status("connecting to " + url + " …");
    state.client = await Client.connect(url);
    state.url = url;
  }
  return state.client;
}

function table(df) {
  if (!df || !df.headers || !df.data || !df.data.length) return "";
  const head = df.headers.map((h) => `<th>${h}</th>`).join("");
  const rows = df.data.map((r) => `<tr>${r.map((c) => `<td>${c ?? ""}</td>`).join("")}</tr>`).join("");
  return `<table><thead><tr>${head}</tr></thead><tbody>${rows}</tbody></table>`;
}

async function analyze() {
  const pair = state.mode === "pair";
  if (!state.vis || (pair && !state.ir)) { status(pair ? "add both images first" : "add a photo first", true); return; }
  $("go").disabled = true;
  try {
    const app = await client();
    status("analyzing …");
    const conf = Number($("conf").value);
    const res = pair ? await app.predict("/analyze_pair", [handle_file(state.vis), handle_file(state.ir), $("model").value, conf])
                     : await app.predict("/analyze_photo", [handle_file(state.vis), conf]);
    const [img, df, msg] = res.data;
    $("board").src = img.url || img;
    $("result").hidden = false;
    $("caption").textContent = msg || "";
    $("log").innerHTML = table(df);
    status(msg || "done");
  } catch (err) {
    console.error(err);
    status("backend unreachable: start demo/app.py or set the backend below (" + (err.message || err) + ")", true);
  } finally {
    $("go").disabled = false;
  }
}

$("go").addEventListener("click", analyze);
renderSamples();
