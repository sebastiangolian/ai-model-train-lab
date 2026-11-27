const $ = (id) => document.getElementById(id);
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
  state: $("state"),
  active: $("active"),
  meta: $("meta"),
  layers: $("layers"),
};
const sliders = ["length", "temperature", "topk"].map((id) => {
  const input = $(id);
  const out = $(`${id}-out`);
  input.addEventListener("input", () => (out.textContent = input.value));
  return input;
});

let generatedNode = null;
let cursorNode = null;
let canvases = [];
let accent = "#c2410c";

const formatBytes = (bytes) => (bytes >= 1e6 ? `${(bytes / 1e6).toFixed(0)} MB` : `${(bytes / 1e3).toFixed(0)} kB`);

function setBusy(busy) {
  ui.generate.disabled = busy;
  ui.stop.disabled = !busy;
}

function buildHeatmaps(layers, neurons) {
  const side = Math.ceil(Math.sqrt(neurons));
  ui.layers.replaceChildren();
  canvases = Array.from({ length: layers }, (_, l) => {
    const figure = document.createElement("figure");
    figure.className = "layer";
    const canvas = document.createElement("canvas");
    canvas.width = canvas.height = side;
    canvas.setAttribute("role", "img");
    canvas.setAttribute("aria-label", `Layer ${l + 1} neuron activity`);
    const caption = document.createElement("figcaption");
    caption.innerHTML = `<span>Layer ${l + 1}</span><span>–</span>`;
    figure.append(canvas, caption);
    ui.layers.append(figure);
    return { canvas, ctx: canvas.getContext("2d"), label: caption.lastElementChild, side };
  });
}

function drawHeatmaps(frames, activity) {
  const [r, g, b] = accent.match(/\w\w/g).map((h) => parseInt(h, 16));
  frames.forEach((frame, l) => {
    const { ctx, side, label } = canvases[l];
    const image = ctx.createImageData(side, side);
    for (let i = 0; i < frame.length; i++) {
      const a = frame[i];
      if (a === 0) continue;
      const p = i * 4;
      image.data[p] = r;
      image.data[p + 1] = g;
      image.data[p + 2] = b;
      image.data[p + 3] = Math.max(70, a);
    }
    ctx.putImageData(image, 0, 0);
    label.textContent = `${(activity[l].xy * 100).toFixed(1)}% on`;
  });
}

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
      const c = meta.config;
      ui.progress.parentElement.hidden = true;
      ui.status.textContent = "Ready";
      ui.state.textContent = formatBytes(meta.stateBytes);
      ui.meta.textContent =
        `${(meta.params / 1e6).toFixed(1)}M params · ${c.n_layer} layers · ${c.n_head} heads × ${meta.neuronsPerHead} neurons · ` +
        `${meta.step} training steps · val ${(meta.valLoss / Math.LN2).toFixed(2)} bits/byte. ` +
        `The state size is fixed: it does not grow with context length.`;
      buildHeatmaps(c.n_layer, c.n_head * meta.neuronsPerHead);
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
      const avg = data.activity.reduce((s, a) => s + a.x, 0) / data.activity.length;
      ui.active.textContent = `${(avg * 100).toFixed(1)}%`;
      drawHeatmaps(data.neurons, data.activity);
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

const scheme = matchMedia("(prefers-color-scheme: dark)");
const readAccent = () => (accent = getComputedStyle(document.documentElement).getPropertyValue("--accent").trim());
scheme.addEventListener("change", readAccent);
readAccent();

worker.postMessage({ type: "load", base: new URL("./model/", import.meta.url).href });
