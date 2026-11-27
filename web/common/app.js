// Shared demo page logic. The page sets <body data-core="./<model>.js"> and serves weights from ./model/.
const $ = (id) => document.getElementById(id);
const pageUrl = new URL(".", location.href);
const worker = new Worker(new URL("./worker.js", import.meta.url), { type: "module" });

const ui = {
  prompt: $("prompt"),
  generate: $("generate"),
  stop: $("stop"),
  status: $("status"),
  progress: $("progress").firstElementChild,
  output: $("output"),
  speed: $("speed"),
  position: $("position"),
  stats: $("stats"),
  meta: $("meta"),
  viz: $("viz"),
};
const sliders = ["length", "temperature", "topk"].map((id) => {
  const input = $(id);
  const out = $(`${id}-out`);
  input.addEventListener("input", () => (out.textContent = input.value));
  return input;
});

let generatedNode = null;
let cursorNode = null;
let modelStats = [];
let renderer = null;
let accent = [194, 65, 12];

const formatBytes = (bytes) => (bytes >= 1e6 ? `${(bytes / 1e6).toFixed(1)} MB` : `${(bytes / 1e3).toFixed(0)} kB`);

function setBusy(busy) {
  ui.generate.disabled = busy;
  ui.stop.disabled = !busy;
}

function updateStats(stats) {
  if (modelStats.length !== stats.length) {
    ui.stats.querySelectorAll(".model-stat").forEach((el) => el.remove());
    modelStats = stats.map(() => {
      const wrap = document.createElement("div");
      wrap.className = "model-stat";
      wrap.innerHTML = "<dt></dt><dd></dd>";
      ui.stats.append(wrap);
      return wrap;
    });
  }
  stats.forEach((stat, i) => {
    const [dt, dd] = modelStats[i].children;
    dt.textContent = stat.label;
    dt.title = stat.title || "";
    dd.textContent = stat.value;
  });
}

// --- Visualisations -------------------------------------------------------

function paint(ctx, width, height, values, { alignRight = false, minAlpha = 0 } = {}) {
  const image = ctx.createImageData(width, height);
  const shift = alignRight ? width * height - values.length : 0;
  for (let i = 0; i < values.length; i++) {
    const a = values[i];
    if (a === 0) continue;
    const p = (i + shift) * 4;
    image.data[p] = accent[0];
    image.data[p + 1] = accent[1];
    image.data[p + 2] = accent[2];
    image.data[p + 3] = Math.max(minAlpha, a);
  }
  ctx.putImageData(image, 0, 0);
}

// BDH: every neuron of a layer as one pixel of a square.
function neuronRenderer(viz) {
  const side = Math.ceil(Math.sqrt(viz.layers[0].length));
  const grid = document.createElement("div");
  grid.className = "layers";
  const cells = viz.layers.map((_, l) => {
    const figure = document.createElement("figure");
    figure.className = "layer";
    const canvas = document.createElement("canvas");
    canvas.width = canvas.height = side;
    canvas.setAttribute("role", "img");
    canvas.setAttribute("aria-label", `Layer ${l + 1} neuron activity`);
    const caption = document.createElement("figcaption");
    caption.innerHTML = `<span>Layer ${l + 1}</span><span></span>`;
    figure.append(canvas, caption);
    grid.append(figure);
    return { ctx: canvas.getContext("2d"), label: caption.lastElementChild };
  });
  ui.viz.replaceChildren(grid);
  return (frame) =>
    frame.layers.forEach((values, l) => {
      paint(cells[l].ctx, side, side, values, { minAlpha: 70 }); // keep sparse spikes visible
      cells[l].label.textContent = frame.captions[l];
    });
}

// GPT: attention of the latest byte over the context window, one row per layer, plus the attended text.
function attentionRenderer(viz) {
  const rows = document.createElement("div");
  rows.className = "attn-rows";
  const lines = viz.rows.map((_, l) => {
    const row = document.createElement("div");
    row.className = "attn-row";
    const canvas = document.createElement("canvas");
    canvas.width = viz.width;
    canvas.height = 1;
    canvas.setAttribute("role", "img");
    canvas.setAttribute("aria-label", `Layer ${l + 1} attention over the context`);
    row.innerHTML = `<span>L${l + 1}</span>`;
    row.append(canvas);
    rows.append(row);
    return canvas.getContext("2d");
  });
  const axis = document.createElement("div");
  axis.className = "attn-axis";
  axis.innerHTML = "<span>← oldest in window</span><span>latest byte →</span>";
  const text = document.createElement("p");
  text.className = "attended";
  ui.viz.replaceChildren(rows, axis, text);
  const decoder = new TextDecoder();
  return (frame) => {
    frame.rows.forEach((values, l) => paint(lines[l], frame.width, 1, values, { alignRight: true }));
    const tail = Math.min(72, frame.tokens.length);
    const spans = [];
    for (let i = frame.tokens.length - tail; i < frame.tokens.length; i++) {
      const span = document.createElement("span");
      const ch = decoder.decode(new Uint8Array([frame.tokens[i]]));
      span.textContent = ch === "\n" ? "↵" : ch;
      span.style.backgroundColor = `rgba(${accent.join(",")}, ${(frame.mean[i] / 255) * 0.85})`;
      spans.push(span);
    }
    text.replaceChildren(...spans);
  };
}

const RENDERERS = { neurons: neuronRenderer, attention: attentionRenderer };

// --- Generation -----------------------------------------------------------

function startGeneration() {
  const prompt = ui.prompt.value;
  ui.output.replaceChildren();
  const promptNode = document.createElement("span");
  promptNode.className = "prompt";
  promptNode.textContent = prompt;
  generatedNode = document.createTextNode("");
  cursorNode = document.createElement("span");
  cursorNode.className = "cursor";
  ui.output.append(promptNode, generatedNode, cursorNode);
  setBusy(true);
  ui.status.textContent = "Reading prompt…";
  worker.postMessage({
    type: "generate",
    prompt,
    maxTokens: Number(sliders[0].value),
    temperature: Number(sliders[1].value),
    topK: Number(sliders[2].value),
  });
}

worker.onmessage = ({ data }) => {
  switch (data.type) {
    case "progress": {
      const pct = data.total ? (100 * data.loaded) / data.total : 0;
      ui.progress.style.width = `${pct}%`;
      ui.status.textContent = `Loading model… ${formatBytes(data.loaded)}${data.total ? ` / ${formatBytes(data.total)}` : ""}`;
      break;
    }
    case "ready": {
      const { meta } = data;
      ui.progress.parentElement.hidden = true;
      ui.status.textContent = "Ready";
      const trained = meta.trainSeconds ? ` in ${Math.round(meta.trainSeconds / 60)} min` : "";
      ui.meta.textContent =
        `${(meta.params / 1e6).toFixed(1)}M params · ${meta.describe} · ${meta.step} training steps${trained} · ` +
        `val ${(meta.valLoss / Math.LN2).toFixed(2)} bits/byte.`;
      setBusy(false);
      break;
    }
    case "prompt":
      ui.status.textContent = `Prompt: ${data.tokens} bytes in ${data.ms.toFixed(0)} ms`;
      break;
    case "tokens": {
      generatedNode.appendData(data.text);
      ui.output.scrollTop = ui.output.scrollHeight;
      ui.speed.textContent = `${data.tokensPerSecond.toFixed(0)} B/s`;
      ui.position.textContent = data.position.toLocaleString();
      updateStats(data.stats);
      renderer ??= RENDERERS[data.viz.kind](data.viz);
      renderer(data.viz);
      break;
    }
    case "done":
      cursorNode?.remove();
      setBusy(false);
      ui.status.textContent = data.stopped ? "Stopped" : "Done";
      break;
    case "error":
      setBusy(false);
      ui.generate.disabled = true;
      ui.status.textContent = `Error: ${data.message}`;
      break;
  }
};

ui.generate.addEventListener("click", startGeneration);
ui.stop.addEventListener("click", () => worker.postMessage({ type: "stop" }));
ui.prompt.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && (e.metaKey || e.ctrlKey) && !ui.generate.disabled) startGeneration();
});
$("chips").addEventListener("click", (e) => {
  const chip = e.target.closest("[data-prompt]");
  if (chip) ui.prompt.value = chip.dataset.prompt;
});

const readAccent = () => {
  const hex = getComputedStyle(document.documentElement).getPropertyValue("--accent").trim();
  accent = hex.match(/\w\w/g).map((h) => parseInt(h, 16));
};
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", readAccent);
readAccent();

worker.postMessage({
  type: "load",
  core: new URL(document.body.dataset.core, pageUrl).href,
  base: new URL("./model/", pageUrl).href,
});
