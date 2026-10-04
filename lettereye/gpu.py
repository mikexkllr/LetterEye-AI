"""GPU detection and model recommendations.

Two things use the GPU:
* Ollama (decision model + OCR LLM) – CUDA / ROCm / Metal / Vulkan, managed by Ollama itself.
* Fast OCR (PP-OCRv6 on ONNX Runtime) – DirectML on Windows, CUDA on Linux, CoreML on macOS.
"""

from __future__ import annotations

import functools
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass

ONNX_DEVICE_LABELS = {
    "dml": "DirectML (GPU)",
    "cuda": "CUDA (NVIDIA GPU)",
    "coreml": "CoreML (Apple GPU/ANE)",
    "cpu": "CPU",
}

_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


@dataclass(frozen=True)
class GpuInfo:
    name: str
    vendor: str
    memory_total_mb: int | None = None
    memory_used_mb: int | None = None

    @property
    def memory_total_gb(self) -> float | None:
        return round(self.memory_total_mb / 1024, 1) if self.memory_total_mb else None


def _run(cmd: list[str], timeout: float = 4.0) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, creationflags=_NO_WINDOW).stdout
    except (OSError, subprocess.SubprocessError):
        return ""


def _nvidia() -> list[GpuInfo]:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return []
    out = _run([exe, "--query-gpu=name,memory.total,memory.used", "--format=csv,noheader,nounits"])
    gpus = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 3:
            try:
                gpus.append(GpuInfo(parts[0], "NVIDIA", int(float(parts[1])), int(float(parts[2]))))
            except ValueError:
                gpus.append(GpuInfo(parts[0], "NVIDIA"))
    return gpus


def _windows_adapters() -> list[GpuInfo]:
    out = _run(["powershell", "-NoProfile", "-Command",
                "Get-CimInstance Win32_VideoController | ForEach-Object { $_.Name }"], timeout=8)
    gpus = []
    for name in (n.strip() for n in out.splitlines()):
        if not name or "Basic Display" in name or "Remote" in name:
            continue
        vendor = "AMD" if ("AMD" in name or "Radeon" in name) else "Intel" if "Intel" in name else \
            "NVIDIA" if "NVIDIA" in name else "GPU"
        gpus.append(GpuInfo(name, vendor))
    return gpus


def _mac() -> list[GpuInfo]:
    if platform.machine() == "arm64":
        mem = _run(["sysctl", "-n", "hw.memsize"]).strip()
        total = int(mem) // (1024 * 1024) if mem.isdigit() else None
        return [GpuInfo("Apple Silicon GPU (unified memory)", "Apple", total)]
    return []


@functools.lru_cache(maxsize=1)
def detect_gpus() -> tuple[GpuInfo, ...]:
    gpus = _nvidia()
    if not gpus and sys.platform == "win32":
        gpus = _windows_adapters()
    if not gpus and sys.platform == "darwin":
        gpus = _mac()
    if not gpus and sys.platform.startswith("linux") and shutil.which("rocm-smi"):
        gpus = [GpuInfo("AMD GPU (ROCm)", "AMD")]
    return tuple(gpus)


def refresh_gpus() -> tuple[GpuInfo, ...]:
    detect_gpus.cache_clear()
    return detect_gpus()


def onnx_providers() -> list[str]:
    try:
        import onnxruntime

        return list(onnxruntime.get_available_providers())
    except Exception:
        return []


def best_onnx_device(preference: str = "auto") -> str:
    """Pick the ONNX Runtime device for fast OCR: 'dml' | 'cuda' | 'coreml' | 'cpu'."""
    if preference == "cpu":
        return "cpu"
    providers = onnx_providers()
    if "CUDAExecutionProvider" in providers and detect_gpus():
        return "cuda"
    if "DmlExecutionProvider" in providers and sys.platform == "win32":
        return "dml"
    if preference == "gpu" and "CoreMLExecutionProvider" in providers and sys.platform == "darwin":
        # CoreML is opt-in: Apple CPUs are already fast for these small models.
        return "coreml"
    return "cpu"


@dataclass(frozen=True)
class ModelChoice:
    name: str
    title: str
    size_gb: float
    min_vram_gb: float
    note: str


DECISION_MODELS = [
    ModelChoice("qwen3.5:2b", "Qwen 3.5 · 2B", 2.7, 4, "Fastest. Good for GPUs with 4 GB."),
    ModelChoice("qwen3.5:4b", "Qwen 3.5 · 4B", 3.4, 6, "Recommended. Best balance of speed and accuracy."),
    ModelChoice("qwen3.5:9b", "Qwen 3.5 · 9B", 6.6, 10, "Most accurate. For GPUs with 10 GB or more."),
    ModelChoice("gemma4:e4b", "Gemma 4 · E4B", 6.6, 10, "Alternative from Google."),
]

OCR_LLM_MODELS = [
    ModelChoice("glm-ocr", "GLM-OCR · 0.9B", 2.2, 3, "Recommended. Top-ranked open OCR model (OmniDocBench)."),
    ModelChoice("deepseek-ocr", "DeepSeek-OCR · 3B", 6.7, 8, "Alternative, larger."),
]


def recommended_models(vram_gb: float | None) -> tuple[str, str]:
    """(decision model, OCR LLM) that fit together in the given VRAM."""
    if vram_gb is None:
        return "qwen3.5:4b", "glm-ocr"
    if vram_gb >= 12:
        return "qwen3.5:9b", "glm-ocr"
    if vram_gb >= 7:
        return "qwen3.5:4b", "glm-ocr"
    return "qwen3.5:2b", "glm-ocr"


def total_vram_gb() -> float | None:
    sizes = [g.memory_total_gb for g in detect_gpus() if g.memory_total_gb]
    return max(sizes) if sizes else None
