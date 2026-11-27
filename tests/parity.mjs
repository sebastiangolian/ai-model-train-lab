// Feed tokens through a browser model core and print per-step logits as JSON (used by the parity tests).
// Usage: node tests/parity.mjs <core module path> <model_dir> <comma-separated tokens>
import { readFileSync } from "node:fs";
import { join, resolve } from "node:path";
import { pathToFileURL } from "node:url";

const [corePath, dir, tokenList] = process.argv.slice(2);
const { loadModel } = await import(pathToFileURL(resolve(corePath)).href);
const manifest = JSON.parse(readFileSync(join(dir, "model.json"), "utf8"));
const bytes = readFileSync(join(dir, "weights.bin"));
const buffer = bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
const state = loadModel(manifest, buffer).newState();
const logits = tokenList.split(",").map((t) => Array.from(state.step(Number(t))));
process.stdout.write(JSON.stringify(logits));
