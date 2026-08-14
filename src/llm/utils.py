import logging

import torch


def setup_logging() -> None:
    # Called in both the server and the worker (a spawned process doesn't
    # inherit logging config). %(processName)s tells the two apart in output.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(processName)s %(name)s: %(message)s",
    )
    # The INFO root level above also unmutes third-party chatter (every HF Hub
    # HTTP request, etc.); keep those at WARNING so our logs stay readable.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("huggingface_hub").setLevel(logging.WARNING)


def get_device() -> str:
    # Prefer CUDA (NVIDIA), then MPS (Apple Silicon), then fall back to CPU.
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
