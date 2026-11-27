/** Sample a token id from logits with temperature and top-k; temperature 0 means greedy. */
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
