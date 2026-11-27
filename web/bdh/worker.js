// Runs the model off the main thread and streams generated text + neuron activity to the page.
import { loadModel, sampleToken } from "./bdh.js";

let model = null;
let stopRequested = false;
const FRAME_MS = 80; // how often to push tokens/activity to the UI

const yieldToEvents = () => new Promise((resolve) => setTimeout(resolve, 0));

async function fetchWithProgress(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`${url}: HTTP ${response.status}`);
  const total = Number(response.headers.get("content-length")) || 0;
  const reader = response.body.getReader();
  const chunks = [];
  let loaded = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    loaded += value.length;
    postMessage({ type: "progress", loaded, total });
  }
  const bytes = new Uint8Array(loaded);
  let offset = 0;
  for (const chunk of chunks) {
    bytes.set(chunk, offset);
    offset += chunk.length;
  }
  return bytes.buffer;
}

// One byte per neuron (0-255, scaled per layer) so a frame stays small.
function neuronFrame(state) {
  return state.neurons.map((layer) => {
    let max = 0;
    for (let i = 0; i < layer.length; i++) if (layer[i] > max) max = layer[i];
    const out = new Uint8Array(layer.length);
    if (max > 0) for (let i = 0; i < layer.length; i++) out[i] = Math.round((255 * layer[i]) / max);
    return out;
  });
}

async function load(base) {
  const manifest = await (await fetch(`${base}model.json`)).json();
  const buffer = await fetchWithProgress(`${base}weights.bin`);
  model = loadModel(manifest, buffer);
  postMessage({
    type: "ready",
    meta: {
      params: manifest.params,
      config: manifest.config,
      step: manifest.step,
      valLoss: manifest.val_loss,
      preset: manifest.preset,
      neuronsPerHead: model.N,
      stateBytes: model.stateBytes,
    },
  });
}

async function generate({ prompt, maxTokens, temperature, topK }) {
  stopRequested = false;
  const state = model.newState();
  const decoder = new TextDecoder();
  const promptBytes = new TextEncoder().encode(prompt || "\n");

  const promptStart = performance.now();
  let logits;
  for (let i = 0; i < promptBytes.length; i++) {
    logits = state.step(promptBytes[i]);
    if (i % 16 === 15) await yieldToEvents();
  }
  postMessage({ type: "prompt", ms: performance.now() - promptStart, tokens: promptBytes.length });

  const started = performance.now();
  let lastFrame = 0;
  let pending = [];
  let generated = 0;
  const flush = (final) => {
    const text = decoder.decode(new Uint8Array(pending), { stream: !final });
    pending = [];
    const frame = neuronFrame(state);
    postMessage(
      {
        type: "tokens",
        text,
        generated,
        position: state.position,
        tokensPerSecond: generated / ((performance.now() - started) / 1000),
        activity: state.activity.map((a) => ({ ...a })),
        neurons: frame,
      },
      frame.map((f) => f.buffer),
    );
  };

  while (generated < maxTokens && !stopRequested) {
    const token = sampleToken(logits, { temperature, topK });
    pending.push(token);
    generated++;
    logits = state.step(token);
    if (performance.now() - lastFrame > FRAME_MS) {
      flush(false);
      lastFrame = performance.now();
      await yieldToEvents(); // lets a "stop" message in
    }
  }
  flush(true);
  postMessage({ type: "done", stopped: stopRequested });
}

onmessage = async ({ data }) => {
  try {
    if (data.type === "load") await load(data.base);
    else if (data.type === "generate") await generate(data);
    else if (data.type === "stop") stopRequested = true;
  } catch (error) {
    postMessage({ type: "error", message: String(error.message || error) });
  }
};
