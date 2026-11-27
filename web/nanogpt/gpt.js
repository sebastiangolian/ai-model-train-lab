// nanoGPT (GPT-2 style Transformer) inference in plain JavaScript with a KV cache.
//
// Each new byte attends (softmax) over the keys/values of every earlier byte in the window, so the
// cache grows by 2 x n_layer x n_embd floats per byte. Learned position embeddings stop at block_size:
// when the window is full we keep the last half of it and re-encode it from position 0 (upstream
// generate() instead re-runs the whole last block_size bytes for every new byte).
// Runs in browsers (Web Worker) and in Node (parity test).
import { formatBytes, layerNorm, readTensors, toBytes } from "../common/tensors.js";

export function loadModel(manifest, buffer) {
  return new GPTModel(manifest.config, manifest.training.block_size, readTensors(manifest, buffer));
}

// Exact GELU (nn.GELU default) needs erf; Numerical Recipes erfc approximation, |error| < 1.2e-7.
function erf(x) {
  const z = Math.abs(x);
  const t = 1 / (1 + 0.5 * z);
  const r =
    t *
    Math.exp(
      -z * z - 1.26551223 +
        t * (1.00002368 + t * (0.37409196 + t * (0.09678418 + t * (-0.18628806 + t * (0.27886807 +
        t * (-1.13520398 + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277)))))))),
    );
  return x >= 0 ? 1 - r : r - 1;
}

// out = W x + b, with W stored (outDim x inDim) row-major like nn.Linear.weight.
function linear(out, W, b, x) {
  const inDim = x.length;
  for (let o = 0; o < out.length; o++) {
    let s = b ? b[o] : 0;
    const row = o * inDim;
    for (let i = 0; i < inDim; i++) s += W[row + i] * x[i];
    out[o] = s;
  }
}

export class GPTModel {
  constructor(config, blockSize, w) {
    this.config = config;
    this.L = config.n_layer;
    this.C = config.n_embd;
    this.H = config.n_head;
    this.T = blockSize;
    this.V = w["transformer.wte.weight"].length / this.C;
    this.wte = w["transformer.wte.weight"]; // V x C, tied with lm_head
    this.wpe = w["transformer.wpe.weight"]; // T x C
    this.lnF = [w["transformer.ln_f.weight"], w["transformer.ln_f.bias"]];
    this.layers = Array.from({ length: this.L }, (_, i) => {
      const p = (name) => w[`transformer.h.${i}.${name}`];
      return {
        ln1: [p("ln_1.weight"), p("ln_1.bias")],
        attn: [p("attn.c_attn.weight"), p("attn.c_attn.bias")],
        proj: [p("attn.c_proj.weight"), p("attn.c_proj.bias")],
        ln2: [p("ln_2.weight"), p("ln_2.bias")],
        fc: [p("mlp.c_fc.weight"), p("mlp.c_fc.bias")],
        fcProj: [p("mlp.c_proj.weight"), p("mlp.c_proj.bias")],
      };
    });
  }

  kvBytes(length) {
    return 2 * this.L * length * this.C * 4;
  }

  describe() {
    return `${this.L} layers · ${this.H} heads · ${this.C}-dim · ${this.T}-byte context window · KV cache grows ${formatBytes(this.kvBytes(1))} per byte`;
  }

  newState() {
    return new GPTState(this);
  }
}

export class GPTState {
  constructor(model) {
    const { L, C, T, V } = model;
    this.model = model;
    this.keys = Array.from({ length: L }, () => new Float32Array(T * C));
    this.values = Array.from({ length: L }, () => new Float32Array(T * C));
    this.length = 0; // bytes currently in the window / KV cache
    this.position = 0; // bytes seen in total
    this.window = []; // tokens currently in the window
    this.refreshes = 0;
    this.x = new Float32Array(C);
    this.h = new Float32Array(C);
    this.qkv = new Float32Array(3 * C);
    this.y = new Float32Array(C);
    this.out = new Float32Array(C);
    this.fc = new Float32Array(4 * C);
    this.scores = new Float32Array(T);
    this.attention = Array.from({ length: L }, () => new Float32Array(T)); // latest byte, mean over heads
    this.logits = new Float32Array(V);
  }

  /** Feed one byte token; returns next-token logits (reused buffer). */
  step(token) {
    if (this.length === this.model.T) this.refresh();
    this.window.push(token);
    this.position++;
    return this.forward(token);
  }

  refresh() {
    const keep = this.window.slice(-Math.floor(this.model.T / 2));
    this.length = 0;
    this.window = keep;
    for (const token of keep) this.forward(token);
    this.refreshes++;
  }

  forward(token) {
    const m = this.model;
    const { C, H, V } = m;
    const hs = C / H;
    const scale = 1 / Math.sqrt(hs);
    const t = this.length;
    const { x, h, qkv, y, out, fc, scores } = this;

    for (let c = 0; c < C; c++) x[c] = m.wte[token * C + c] + m.wpe[t * C + c];

    m.layers.forEach((layer, l) => {
      h.set(x);
      layerNorm(h, ...layer.ln1);
      linear(qkv, ...layer.attn, h);
      const K = this.keys[l];
      const Vc = this.values[l];
      K.set(qkv.subarray(C, 2 * C), t * C);
      Vc.set(qkv.subarray(2 * C, 3 * C), t * C);

      const attention = this.attention[l];
      attention.fill(0);
      y.fill(0);
      for (let head = 0; head < H; head++) {
        const off = head * hs;
        let max = -Infinity;
        for (let s = 0; s <= t; s++) {
          let dot = 0;
          for (let j = 0; j < hs; j++) dot += qkv[off + j] * K[s * C + off + j];
          scores[s] = dot * scale;
          if (scores[s] > max) max = scores[s];
        }
        let sum = 0;
        for (let s = 0; s <= t; s++) sum += scores[s] = Math.exp(scores[s] - max);
        for (let s = 0; s <= t; s++) {
          const weight = scores[s] / sum;
          attention[s] += weight / H;
          for (let j = 0; j < hs; j++) y[off + j] += weight * Vc[s * C + off + j];
        }
      }
      linear(out, ...layer.proj, y);
      for (let c = 0; c < C; c++) x[c] += out[c];

      h.set(x);
      layerNorm(h, ...layer.ln2);
      linear(fc, ...layer.fc, h);
      for (let i = 0; i < fc.length; i++) fc[i] = 0.5 * fc[i] * (1 + erf(fc[i] / Math.SQRT2));
      linear(out, ...layer.fcProj, fc);
      for (let c = 0; c < C; c++) x[c] += out[c];
    });

    layerNorm(x, ...m.lnF);
    for (let v = 0; v < V; v++) {
      let s = 0;
      const row = v * C;
      for (let c = 0; c < C; c++) s += m.wte[row + c] * x[c];
      this.logits[v] = s;
    }
    this.length++;
    return this.logits;
  }

  stats() {
    return [
      { label: "KV cache", value: formatBytes(this.model.kvBytes(this.length)), title: "Grows with every byte in the context window" },
      {
        label: "Window",
        value: `${this.length}/${this.model.T}`,
        title: `Bytes the model can see. Re-encoded ${this.refreshes}× after filling up`,
      },
    ];
  }

  /** Attention of the latest byte over the window, per layer (mean over heads). */
  viz() {
    const rows = this.attention.map((a) => toBytes(a.subarray(0, this.length)));
    const mean = new Float32Array(this.length);
    for (const a of this.attention) for (let s = 0; s < this.length; s++) mean[s] += a[s];
    return { kind: "attention", rows, mean: toBytes(mean), tokens: this.window.slice(), width: this.model.T };
  }
}
