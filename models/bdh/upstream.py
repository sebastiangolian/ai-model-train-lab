"""Import the official BDH implementation from the pinned upstream checkout in vendor/bdh."""

import sys
from pathlib import Path

VENDOR_DIR = Path(__file__).resolve().parents[2] / "vendor" / "bdh"

if not (VENDOR_DIR / "bdh.py").exists():
    raise ImportError(
        f"Upstream BDH code not found in {VENDOR_DIR}. Run `make setup MODEL=bdh` first."
    )

if str(VENDOR_DIR) not in sys.path:
    sys.path.insert(0, str(VENDOR_DIR))

import bdh  # noqa: E402  (upstream module: BDH, BDHConfig)

BDH = bdh.BDH
BDHConfig = bdh.BDHConfig

__all__ = ["BDH", "BDHConfig", "VENDOR_DIR"]
