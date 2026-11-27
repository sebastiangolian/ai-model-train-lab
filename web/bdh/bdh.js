// BDH (Dragon Hatchling) inference in plain JavaScript, in its recurrent form.
//
// Upstream bdh.py runs attention in parallel over the whole sequence:
//   yKV_t = sum_{s<t} (q_t . q_s) x_s            (linear attention, no softmax, Q = K)
// which equals q_t^T S_t with a fixed-size "synaptic" state S_t = sum_{s<t} q_s x_s^T (N x D per head).
// So generation needs O(1) memory per token instead of a growing KV cache, and because q is a
// sparse non-negative vector, both reading and updating S only touch the active neurons.
// Runs in browsers (Web Worker) and in Node (parity test).
import { formatBytes, layerNorm, readTensors, toBytes } from "../common/tensors.js";

const TWO_PI = 2 * Math.PI;

/** Build a model from model.json (parsed) and weights.bin (ArrayBuffer). */
export function loadModel(manifest, buffer) {
  return new BDHModel(manifest.config, readTensors(manifest, buffer));
}

// out[n] = relu(sum_d x[d] * W[base + d*N + n]) for a (D x N) slice of W; returns active count.
function sparseProject(x, W, base, N, out) {
  out.fill(0);
  for (let d = 0; d < x.length; d++) {
    const xd = x[d];
    if (xd === 0) continue;
    const row = base + d * N;
    for (let n = 0; n < N; n++) out[n] += xd * W[row + n];
  }
  let active = 0;
  for (let n = 0; n < N; n++) {
    if (out[n] > 0) active++;
    else out[n] = 0;
  }
  return active;
}

export class BDHModel {
  constructor(config, weights) {
    this.config = config;
    this.L = config.n_layer;
    this.D = config.n_embd;
    this.H = config.n_head;
    this.N = (config.mlp_internal_dim_multiplier * config.n_embd) / config.n_head;
    this.V = config.vocab_size;
    this.embed = weights["embed.weight"]; // V x D
    this.encoder = weights["encoder"]; // H x D x N
    this.encoderV = weights["encoder_v"]; // H x D x N
    this.decoder = weights["decoder"]; // (H*N) x D
    this.lmHead = weights["lm_head"]; // D x V
    this.freqs = weights["attn.freqs"]; // N
  }

  /** Bytes of synaptic state, constant no matter how long the context gets. */
  get stateBytes() {
    return this.L * this.H * this.N * this.D * 4;
  }

  describe() {
    return `${this.L} layers · ${this.H} heads × ${this.N} neurons · fixed synaptic state of ${formatBytes(this.stateBytes)}`;
  }

  newState() {
    return new BDHState(this);
  }
}

export class BDHState {
  constructor(model) {
    const { L, H, N, D, V } = model;
    this.model = model;
    this.position = 0;
    this.synapses = new Float32Array(L * H * N * D);
    this.x = new Float32Array(D);
    this.xSparse = new Float32Array(N);
    this.q = new Float32Array(N);
    this.ySparse = new Float32Array(N);
    this.yKV = new Float32Array(D);
    this.yMLP = new Float32Array(D);
    this.cos = new Float32Array(N);
    this.sin = new Float32Array(N);
    this.logits = new Float32Array(V);
    // Per layer: fraction of active neurons (x after ReLU, and the gated x*y product).
    this.activity = Array.from({ length: L }, () => ({ x: 0, xy: 0 }));
    // Per layer, per neuron (H*N): latest gated activation, for the heatmap.
    this.neurons = Array.from({ length: L }, () => new Float32Array(H * N));
  }

  /** Feed one byte token; returns next-token logits (reused buffer). */
  step(token) {
    const m = this.model;
    const { L, H, N, D, V } = m;
    const { x, xSparse, q, ySparse, yKV, yMLP, cos, sin, synapses } = this;

    x.set(m.embed.subarray(token * D, (token + 1) * D));
    layerNorm(x);

    // RoPE phases for this position (same float32 arithmetic as upstream).
    for (let n = 0; n < N; n++) {
      const phase = (Math.fround(this.position * m.freqs[n]) % 1) * TWO_PI;
      cos[n] = Math.cos(phase);
      sin[n] = Math.sin(phase);
    }

    for (let l = 0; l < L; l++) {
      yMLP.fill(0);
      let activeX = 0;
      let activeXY = 0;
      const neurons = this.neurons[l];
      for (let h = 0; h < H; h++) {
        activeX += sparseProject(x, m.encoder, h * D * N, N, xSparse);

        // q = rope(x_sparse); zero pairs stay zero, so q is as sparse as x_sparse.
        for (let i = 0; i < N; i += 2) {
          const a = xSparse[i];
          const b = xSparse[i + 1];
          q[i] = a * cos[i] - b * sin[i];
          q[i + 1] = b * cos[i + 1] + a * sin[i + 1];
        }

        // Read the past (strictly earlier tokens), then write the current token into the synapses.
        yKV.fill(0);
        const S = (l * H + h) * N * D;
        for (let n = 0; n < N; n++) {
          const qn = q[n];
          if (qn === 0) continue;
          const row = S + n * D;
          for (let d = 0; d < D; d++) {
            yKV[d] += qn * synapses[row + d];
            synapses[row + d] += qn * x[d];
          }
        }
        layerNorm(yKV);

        sparseProject(yKV, m.encoderV, h * D * N, N, ySparse);
        for (let n = 0; n < N; n++) {
          const xy = xSparse[n] * ySparse[n];
          neurons[h * N + n] = xy;
          if (xy === 0) continue;
          activeXY++;
          const row = (h * N + n) * D;
          for (let d = 0; d < D; d++) yMLP[d] += xy * m.decoder[row + d];
        }
      }
      layerNorm(yMLP);
      for (let d = 0; d < D; d++) x[d] += yMLP[d];
      layerNorm(x);
      this.activity[l].x = activeX / (H * N);
      this.activity[l].xy = activeXY / (H * N);
    }

    const logits = this.logits;
    logits.fill(0);
    for (let d = 0; d < D; d++) {
      const xd = x[d];
      const row = d * V;
      for (let v = 0; v < V; v++) logits[v] += xd * m.lmHead[row + v];
    }
    this.position++;
    return logits;
  }

  stats() {
    const firing = this.activity.reduce((s, a) => s + a.x, 0) / this.activity.length;
    return [
      { label: "Synaptic state", value: formatBytes(this.model.stateBytes), title: "Fixed size, independent of context length" },
      { label: "Firing (x > 0)", value: `${(firing * 100).toFixed(1)}%`, title: "Share of neurons active after ReLU, averaged over layers" },
    ];
  }

  /** Every neuron of every layer for the latest byte (gated activation x·y). */
  viz() {
    return {
      kind: "neurons",
      layers: this.neurons.map(toBytes),
      captions: this.activity.map((a) => `${(a.xy * 100).toFixed(1)}% on`),
    };
  }
}
