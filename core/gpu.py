"""
GPU auto-detection, CUDA force, and Ollama LLM factory.
"""

import os

from langchain_ollama import OllamaLLM
from rich.console import Console

console = Console()

# Module-level device tracker (mutable via set_device / force_gpu)
DEVICE: str = "cpu"


def set_device(device: str) -> None:
    """Set the global device string."""
    global DEVICE
    DEVICE = device


def get_device() -> str:
    """Return the current device string."""
    return DEVICE


def force_gpu(device_id: int = 0) -> str:
    """
    Force CUDA GPU usage. Sets env vars for Ollama and sentence-transformers.
    Returns 'cuda:N' or 'cpu'.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
    os.environ["CUDA_DEVICE_ORDER"]    = "PCI_BUS_ID"
    os.environ["OLLAMA_NUM_GPU"]       = "99"
    os.environ["OLLAMA_GPU_LAYERS"]    = "99"

    try:
        import torch
        if not torch.cuda.is_available():
            console.print(
                "[red bold]CUDA not available to torch.\n"
                "[yellow]Check: nvidia-smi | nvcc --version | "
                "pip install torch --index-url https://download.pytorch.org/whl/cu121"
            )
            return "cpu"

        device = f"cuda:{device_id}"
        try:
            _ = torch.zeros(1, device=device)
        except RuntimeError as e:
            console.print(f"[red]GPU warm-up failed on {device}: {e}")
            return "cpu"

        name    = torch.cuda.get_device_name(device_id)
        mem     = torch.cuda.get_device_properties(device_id).total_mem // (1024 ** 2)
        compute = torch.cuda.get_device_capability(device_id)
        console.print(
            f"  [green bold]GPU FORCED: {name}  |  {mem} MB VRAM  |  "
            f"Compute {compute[0]}.{compute[1]}  |  device={device} ✓"
        )
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        return device

    except ImportError:
        console.print(
            "[red]torch not installed.\n"
            "[yellow]Run: pip install torch --index-url https://download.pytorch.org/whl/cu121"
        )
        return "cpu"


def get_llm(model: str) -> OllamaLLM:
    """Return an Ollama LLM instance, offloading to GPU when available."""
    on_gpu = DEVICE.startswith("cuda") or DEVICE == "mps"
    return OllamaLLM(model=model, temperature=0, num_gpu=-1 if on_gpu else 0)
