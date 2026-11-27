import torch


def pick_device(name: str = "auto") -> torch.device:
    """Return the requested device, or the best available one for "auto" (cuda > mps > cpu)."""
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def synchronize(device: torch.device) -> None:
    """Wait for queued kernels so wall-clock timings are honest."""
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def describe_hardware(device: torch.device) -> str:
    """Human-readable accelerator name, e.g. "Apple M4 Pro (mps)"."""
    import platform
    import subprocess

    if device.type == "cuda":
        return f"{torch.cuda.get_device_name(device)} (cuda)"
    name = platform.processor() or platform.machine()
    if platform.system() == "Darwin":
        try:
            name = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True, check=True
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            pass
    return f"{name} ({device.type})"
