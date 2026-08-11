"""GPU / device detection and introspection.

Two responsibilities:
  1. Pick a device string ('cuda:0', 'cpu', ...) at process startup.
  2. Report GPU state to the /gpu endpoint so students can verify their
     Coolify GPU config is working.

Kept in its own module so main.py doesn't need to know anything about
torch or CUDA. If we ever swap PyTorch for JAX or something else, this
module is what changes.
"""
import os


def detect_device() -> str:
    """Pick the best available device for the HF pipeline.

    Precedence:
      1. If DEVICE env var is set, use it verbatim (deterministic override).
      2. If CUDA is available and multiple GPUs are visible, pick the one
         with the most free VRAM (least-loaded auto-pick). Handles the case
         where several student containers share a machine and see all GPUs.
      3. If exactly one GPU is visible (e.g., because the container was
         started with CUDA_VISIBLE_DEVICES=N), use cuda:0.
      4. Otherwise, fall back to CPU.

    Admin-side alternative: at Coolify Application creation, set
    CUDA_VISIBLE_DEVICES=<group_num % 4> on each container to pin it to a
    specific GPU. Cheaper than runtime auto-pick and avoids race conditions
    when multiple containers restart simultaneously.
    """
    if forced := os.environ.get("DEVICE"):
        return forced
    try:
        import torch

        if not torch.cuda.is_available():
            return "cpu"
        n = torch.cuda.device_count()
        if n == 1:
            return "cuda:0"
        best_idx = max(
            range(n),
            key=lambda i: torch.cuda.mem_get_info(i)[0],  # free bytes on device i
        )
        return f"cuda:{best_idx}"
    except ImportError:
        pass
    return "cpu"


# Resolved once at import time. Modules that need it (local_classifier,
# main) import DEVICE directly.
DEVICE = detect_device()


def gpu_status() -> dict:
    """Return structured GPU-state info for the /gpu endpoint.

    Reports whether torch is installed, whether CUDA is available, and
    per-GPU name / total VRAM / allocated VRAM / free VRAM.
    """
    info: dict = {
        "device_setting": DEVICE,
        "using_gpu": DEVICE.startswith("cuda"),
        "torch_installed": False,
        "cuda_available": False,
        "device_count": 0,
        "devices": [],
    }
    try:
        import torch

        info["torch_installed"] = True
        info["torch_version"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["device_count"] = torch.cuda.device_count()
            info["cuda_version"] = torch.version.cuda
            info["devices"] = [
                {
                    "index": i,
                    "name": torch.cuda.get_device_name(i),
                    "memory_total_gb": round(
                        torch.cuda.get_device_properties(i).total_memory / (1024**3), 1
                    ),
                    "memory_allocated_gb": round(
                        torch.cuda.memory_allocated(i) / (1024**3), 2
                    ),
                    "memory_free_gb": round(
                        torch.cuda.mem_get_info(i)[0] / (1024**3), 2
                    ),
                }
                for i in range(torch.cuda.device_count())
            ]
    except ImportError:
        pass
    return info
