"""Model registry: every model lives in models/<name>/spec.py and exposes the same small interface.

    NAME: str                       folder name under models/, runs/ and web/
    DEFAULT_PRESET: str
    PRESETS: dict[str, TrainConfig]
    FLOAT32_TENSORS: set[str]       state_dict entries exported as float32 instead of float16
    build_model(config, device)     returns an nn.Module with forward(idx, targets) -> (logits, loss)
                                    and generate(idx, max_new_tokens, temperature, top_k)
"""

import importlib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
VENDOR_DIR = ROOT / "vendor"
RUNS_DIR = ROOT / "runs"
WEB_DIR = ROOT / "web"


def available_models() -> list[str]:
    return sorted(p.parent.name for p in MODELS_DIR.glob("*/spec.py"))


def load_spec(name: str) -> ModuleType:
    if name not in available_models():
        raise SystemExit(f"Unknown model {name!r}. Available: {', '.join(available_models())}")
    return importlib.import_module(f"models.{name}.spec")


def load_vendor_module(vendor: str, filename: str) -> ModuleType:
    """Import an upstream file from vendor/<vendor>/ under a unique module name (e.g. nanoGPT's model.py)."""
    module_name = f"vendor_{vendor}_{Path(filename).stem}"
    if module_name in sys.modules:
        return sys.modules[module_name]
    path = VENDOR_DIR / vendor / filename
    if not path.exists():
        raise ImportError(f"Upstream code not found at {path}. Run `make setup MODEL={vendor}` first.")
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def upstream_ref(vendor: str) -> str:
    head = VENDOR_DIR / vendor / ".git" / "HEAD"
    return head.read_text().strip() if head.exists() else "unknown"
