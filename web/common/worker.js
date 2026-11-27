// Runs a model core off the main thread and streams generated text, stats and visualisation frames.
// The core module (e.g. ../bdh/bdh.js) exports loadModel(manifest, buffer) -> model with
// describe() and newState(); a state has step(token) -> logits, position, stats() and viz().
import { sampleToken } from "./sampling.js";

let model = null;
let stopRequested = false;
const FRAME_MS = 80; // how often to push tokens/visualisation to the UI

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

async function load({ core, base }) {
  const { loadModel } = await import(core);
  const manifest = await (await fetch(`${base}model.json`)).json();
  const buffer = await fetchWithProgress(`${base}weights.bin`);
  model = loadModel(manifest, buffer);
  postMessage({
    type: "ready",
    meta: {
      params: manifest.params,
      step: manifest.step,
      valLoss: manifest.val_loss,
      trainSeconds: manifest.train_seconds,
      device: manifest.device,
      describe: model.describe(),
    },
  });
}

function transferables(viz) {
  return [...(viz.layers || []), ...(viz.rows || []), ...(viz.mean ? [viz.mean] : [])].map((a) => a.buffer);
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
    const viz = state.viz();
    postMessage(
      {
        type: "tokens",
        text,
        generated,
        position: state.position,
        tokensPerSecond: generated / ((performance.now() - started) / 1000),
        stats: state.stats(),
        viz,
      },
      transferables(viz),
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
    if (data.type === "load") await load(data);
    else if (data.type === "generate") await generate(data);
    else if (data.type === "stop") stopRequested = true;
  } catch (error) {
    postMessage({ type: "error", message: String(error.message || error) });
  }
};
