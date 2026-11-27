// Read tensors exported by lab/export.py (model.json + weights.bin) into Float32Arrays.

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

/** Returns {name: Float32Array}; tied weights ("alias") share the same array. */
export function readTensors(manifest, buffer) {
  const weights = {};
  const entries = Object.entries(manifest.tensors);
  for (const [name, spec] of entries) if (!spec.alias) weights[name] = readTensor(buffer, spec);
  for (const [name, spec] of entries) if (spec.alias) weights[name] = weights[spec.alias];
  return weights;
}

export function layerNorm(v, weight = null, bias = null) {
  let mean = 0;
  for (let i = 0; i < v.length; i++) mean += v[i];
  mean /= v.length;
  let variance = 0;
  for (let i = 0; i < v.length; i++) variance += (v[i] - mean) ** 2;
  const inv = 1 / Math.sqrt(variance / v.length + 1e-5);
  for (let i = 0; i < v.length; i++) {
    let y = (v[i] - mean) * inv;
    if (weight) y = y * weight[i] + (bias ? bias[i] : 0);
    v[i] = y;
  }
}

/** One byte per value (0-255), scaled to the array's max, so frames sent to the UI stay small. */
export function toBytes(values) {
  let max = 0;
  for (let i = 0; i < values.length; i++) if (values[i] > max) max = values[i];
  const out = new Uint8Array(values.length);
  if (max > 0) for (let i = 0; i < values.length; i++) out[i] = Math.round((255 * values[i]) / max);
  return out;
}

export const formatBytes = (bytes) =>
  bytes >= 1e6 ? `${(bytes / 1e6).toFixed(1)} MB` : `${(bytes / 1e3).toFixed(0)} kB`;
