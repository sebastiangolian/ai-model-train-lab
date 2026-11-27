// BDH (Dragon Hatchling) inference in plain JavaScript, in its recurrent form.
//
// Upstream bdh.py runs attention in parallel over the whole sequence:
//   yKV_t = sum_{s<t} (q_t . q_s) x_s            (linear attention, no softmax, Q = K)
// which equals q_t^T S_t with a fixed-size "synaptic" state S_t = sum_{s<t} q_s x_s^T (N x D per head).
// So generation needs O(1) memory per token instead of a growing KV cache, and because q is a
// sparse non-negative vector, both reading and updating S only touch the active neurons.
// Runs in browsers (Web Worker) and in Node (parity test).

const LN_EPS = 1e-5;
const TWO_PI = 2 * Math.PI;

const HALF_TO_FLOAT = (() => {
  const table = new Float32Array(65536);
  for (let h = 0; h < 65536; h++) {
    const sign = h & 0x8000 ? -1 : 1;
    const exp = (h >> 10) & 0x1f;
    const mant = h & 0x3ff;
    if (exp === 0) table[h] = sign * mant * 2 ** -24;
    else if (exp === 31) table[h] = mant ? NaN : sign * Infinity;
    else table[h] = sign * (1 + mant / 1024) * 2 ** (exp - 15);
  }
  return table;
})();

function readTensor(buffer, { dtype, shape, offset }) {
  const count = shape.reduce((a, b) => a * b, 1);
  if (dtype === "float32") return new Float32Array(buffer.slice(offset, offset + count * 4));
  const half = new Uint16Array(buffer, offset, count);
  const out = new Float32Array(count);
  for (let i = 0; i < count; i++) out[i] = HALF_TO_FLOAT[half[i]];
  return out;
}

/** Build a model from model.json (parsed) and weights.bin (ArrayBuffer). */
export function loadModel(manifest, buffer) {
  const weights = {};
  for (const [name, spec] of Object.entries(manifest.tensors)) weights[name] = readTensor(buffer, spec);
  return new BDHModel(manifest.config, weights);
}

function layerNorm(v) {
  let mean = 0;
  for (let i = 0; i < v.length; i++) mean += v[i];
  mean /= v.length;
  let variance = 0;
  for (let i = 0; i < v.length; i++) variance += (v[i] - mean) ** 2;
  const inv = 1 / Math.sqrt(variance / v.length + LN_EPS);
  for (let i = 0; i < v.length; i++) v[i] = (v[i] - mean) * inv;
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
}

/** Sample a token id; temperature 0 means greedy. */
export function sampleToken(logits, { temperature = 0.8, topK = 20 } = {}, random = Math.random) {
  const V = logits.length;
  if (temperature <= 0) {
    let best = 0;
    for (let v = 1; v < V; v++) if (logits[v] > logits[best]) best = v;
    return best;
  }
  const order = Array.from({ length: V }, (_, v) => v).sort((a, b) => logits[b] - logits[a]);
  const k = Math.min(Math.max(1, topK || V), V);
  const top = order.slice(0, k);
  const max = logits[top[0]];
  const weights = top.map((v) => Math.exp((logits[v] - max) / temperature));
  let r = random() * weights.reduce((a, b) => a + b, 0);
  for (let i = 0; i < k; i++) {
    r -= weights[i];
    if (r <= 0) return top[i];
  }
  return top[k - 1];
}
