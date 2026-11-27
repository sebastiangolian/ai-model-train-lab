# ai-model-train-lab

New AI architectures, trained from scratch on a laptop and run in the browser. Every model shares the same data,
training recipe and `make` pipeline (setup, train, evaluate, checkpoint, sample, export, compare), so results are
directly comparable, and gets a client-side demo on GitHub Pages.

**Live demos:** https://sebastiangolian.github.io/ai-model-train-lab/

| Model | What | Demo |
| --- | --- | --- |
| [`bdh`](#bdh--dragon-hatchling) | Pathway's Dragon Hatchling, a "post-Transformer" architecture | [/bdh/](https://sebastiangolian.github.io/ai-model-train-lab/bdh/) |
| [`nanogpt`](#nanogpt--transformer-baseline) | karpathy/nanoGPT, a GPT-2 style Transformer: the baseline | [/nanogpt/](https://sebastiangolian.github.io/ai-model-train-lab/nanogpt/) |

## Results

Tiny Shakespeare, shared recipe (below), Apple M4 Pro (PyTorch MPS). Best validation checkpoint; lower is better.

| Model | Preset | Params | Steps | Bytes seen | Val bits/byte | Train time |
| --- | --- | --- | --- | --- | --- | --- |
| BDH | `small` | 12.7M | 1,500 | 6.1M | **2.20** | 20.3 min |
| nanoGPT | `small` (same data) | 12.6M | 1,500 | 6.1M | 2.80 | 2.9 min |
| nanoGPT | `small-long` (same time) | 12.6M | 10,000 (best @7,500) | 41.0M | **2.18** | 18.5 min |

- **Same data:** BDH learns much more per byte (2.20 vs 2.80).
- **Same time:** a tie (2.18 vs 2.20, within noise): a Transformer step is ~7× cheaper here, so it sees ~7× more data.
- Both start to overfit the 1 MB corpus. Caveats: one seed per run, MPS on a laptop (BDH's large sparse layers may
  fare differently on CUDA), one shared recipe without per-model tuning (e.g. no warmup for the Transformer).

## Quick start

```bash
make all MODEL=bdh            # venv + deps, pinned upstream code, dataset, train, sample, export to web/bdh/
make all MODEL=nanogpt
make compare                  # table of all runs, also written to web/results.json
make web                      # serve the demos on http://localhost:8000
```

Requires Python 3.13 (`PYTHON=...` to override) and git. `node` is only needed for the JS parity tests.

| Target | What it does |
| --- | --- |
| `make setup` | Create `.venv`, install `requirements.txt`, run the model's `model-setup` (upstream code, dataset) |
| `make train` | Train `MODEL`/`PRESET` into `runs/MODEL/PRESET/` (`checkpoint.pt` = best val, `metrics.csv`, `config.json`, `run.json`, `sample.txt`) |
| `make sample` | Generate from a checkpoint, e.g. `make sample MODEL=nanogpt PROMPT='JULIET:\n' TOKENS=600 SAMPLES=3` |
| `make export` | Convert the checkpoint into `web/MODEL/model/` for the browser demo |
| `make compare` | Compare all runs (params, bytes seen, best val bits/byte, train time) |
| `make web` | Serve `web/` locally |
| `make test` | Shared tests + `tests/MODEL` |
| `make models` | List available models |
| `make clean` / `make distclean` | Remove the model's runs / everything generated |

Common variables: `MODEL`, `PRESET`, `DEVICE=auto|cpu|mps|cuda`, `ITERS=N` (override iterations).
Ctrl+C during training keeps the best checkpoint saved so far.

### Shared recipe

Tiny Shakespeare as raw bytes (vocab 256, 90/10 train/val split), AdamW with constant lr 1e-3 and weight decay 0.1,
dropout 0.1, batch 16 × 256 bytes. Presets of different models are parameter-matched (`tiny` ≈ 1.7M, `small` ≈ 12.7M,
`full` ≈ 25M). Validation loss is reported in bits per byte; the checkpoint keeps the best validation step.

### Adding a model

1. `models/<name>/spec.py`: `NAME`, `DEFAULT_PRESET`, `PRESETS` (of `lab.config.TrainConfig`), `FLOAT32_TENSORS`,
   `build_model(config, device)` returning a module with `forward(idx, targets) -> (logits, loss)` and
   `generate(idx, max_new_tokens, temperature, top_k)`.
2. `models/<name>/model.mk`: defaults (`PRESET`, `PROMPT`) and a `model-setup` target (e.g. clone upstream into `vendor/<name>`).
3. `tests/<name>/`: use `tests.conftest.assert_trains_and_generates` and `assert_js_matches_pytorch`.
4. For a demo: `web/<name>/<core>.js` exporting `loadModel(manifest, buffer)` (see `web/common/worker.js` for the
   interface), `web/<name>/index.html` with `<body data-core="./<core>.js">`, and a card in `web/index.html`.

## BDH — Dragon Hatchling

Wrapper around [pathwaycom/bdh](https://github.com/pathwaycom/bdh), the official toy implementation published with
the paper [The Dragon Hatchling: The Missing Link between the Transformer and Models of the Brain](https://arxiv.org/abs/2509.26507).

> Scope: this is the small byte-level language model from the paper's repository. It is **not** BDH-CQ and does
> not reproduce Pathway's reasoning results (ARC-AGI-1, Sudoku Extreme); those models are not public.

Upstream BDH computes attention in parallel over the whole sequence, without softmax and with `Q = K`:

```
yKV_t = Σ_{s<t} (q_t · q_s) x_s  =  q_tᵀ S_t,     S_t = Σ_{s<t} q_s x_sᵀ   (N × D per head)
```

`web/bdh/bdh.js` uses the right-hand side: a fixed-size "synaptic" state `S` updated once per byte, instead of
re-running the whole prompt for every new token (what upstream `generate()` does) or keeping a growing KV cache.
Because `q` is sparse and non-negative (ReLU), reading and updating `S` only touches the active neurons. The demo
shows every neuron's activation per layer.

## nanoGPT — Transformer baseline

[karpathy/nanoGPT](https://github.com/karpathy/nanoGPT) `model.py` at a pinned commit, configured for bytes and
parameter-matched to BDH. Presets: `small` has the same data budget as BDH `small` (1,500 steps); `small-long` has
roughly the same training time on Apple M4 Pro (10,000 steps), because a Transformer step is ~7× cheaper here.

`web/nanogpt/gpt.js` decodes with a KV cache and shows where the latest byte attends. Learned position embeddings stop
at 256 bytes: when the window is full, the last 128 bytes are re-encoded from position 0 (upstream `generate()`
re-runs the last 256 bytes for every new byte instead).

## How the demos are verified

The parity tests export a random model, run the same bytes through PyTorch (whole sequence in parallel) and through
the JavaScript core in Node (one byte at a time), and require matching logits (max difference ~1e-7). For nanoGPT a
second test covers the window re-encoding.

## Layout

```
lab/              shared: data, training loop, sampling, export, comparison, model registry
models/<name>/    spec.py (presets + build_model) and model.mk (upstream code, data)
tests/            shared helpers and tests, tests/<name>/ per model
web/              index.html (demos + results), common/ (UI, worker, tensors), <name>/ (core JS, page, exported weights)
runs/, data/, vendor/   generated by make, not committed
```

The site is deployed from `web/` by `.github/workflows/deploy-pages.yml` on every push to `main`. To publish a newly
trained model: `make train export MODEL=<name>`, `make compare`, then commit `web/`.

Upstream code (MIT: Pathway Technology, Inc.; Andrej Karpathy) is cloned into `vendor/` and not copied here.
