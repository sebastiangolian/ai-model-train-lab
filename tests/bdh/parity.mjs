// Feed tokens through web/bdh/bdh.js and print logits as JSON (used by test_web_parity.py).
// Usage: node tests/bdh/parity.mjs <model_dir> <comma-separated tokens>
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { loadModel } from "../../web/bdh/bdh.js";

const [dir, tokenList] = process.argv.slice(2);
const manifest = JSON.parse(readFileSync(join(dir, "model.json"), "utf8"));
const bytes = readFileSync(join(dir, "weights.bin"));
const buffer = bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
const state = loadModel(manifest, buffer).newState();
const logits = tokenList.split(",").map((t) => Array.from(state.step(Number(t))));
process.stdout.write(JSON.stringify(logits));
