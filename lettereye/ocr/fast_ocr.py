"""Fast OCR with PP-OCRv6 (RapidOCR on ONNX Runtime), GPU-accelerated where available.

DirectML on Windows (any DX12 GPU: NVIDIA, AMD, Intel), CUDA on Linux, optional CoreML on macOS,
otherwise CPU. Models (~30 MB for 'small') are downloaded once into the app-data folder.
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .. import paths
from ..gpu import best_onnx_device
from . import OcrResult

log = logging.getLogger(__name__)

# PP-OCRv6 is one multilingual model; RapidOCR validates the language code against this list.
PP_OCRV6_LANGS = {
    "en", "de", "fr", "es", "it", "pt", "nl", "pl", "cs", "sk", "sl", "hr", "bs", "hu", "ro", "da", "no", "sv",
    "fi", "et", "lt", "lv", "is", "ga", "cy", "tr", "id", "ms", "vi", "sq", "ca", "eu", "gl", "lb", "mt",
}


class FastOCR:
    def __init__(self, size: str = "small", device: str = "auto", language: str = "de",
                 model_dir: Path | None = None):
        self.size = size
        self.preference = device
        self.language = language if language in PP_OCRV6_LANGS else "en"
        self.model_dir = model_dir or paths.ocr_models_dir()
        self._engine = None
        self._lock = threading.Lock()
        self.device = best_onnx_device(device)
        self.active_provider = ""

    def _build(self):
        from rapidocr import ModelType, OCRVersion, RapidOCR

        params = {
            "Global.log_level": "warning",
            "Global.model_root_dir": str(self.model_dir),
            "Det.ocr_version": OCRVersion.PPOCRV6,
            "Det.model_type": ModelType(self.size),
            "Det.lang_type": self.language,
            "Rec.ocr_version": OCRVersion.PPOCRV6,
            "Rec.model_type": ModelType(self.size),
            "Rec.lang_type": self.language,
            "EngineConfig.onnxruntime.use_cuda": self.device == "cuda",
            "EngineConfig.onnxruntime.use_dml": self.device == "dml",
            "EngineConfig.onnxruntime.use_coreml": self.device == "coreml",
        }
        if self.device == "coreml":
            params["EngineConfig.onnxruntime.coreml_ep_cfg.ModelCacheDirectory"] = str(self.model_dir / "coreml-cache")
        last_error: Exception | None = None
        for attempt in range(4):  # first run downloads models; retry flaky connections
            try:
                engine = RapidOCR(params=params)
                break
            except Exception as exc:  # DownloadFileException and friends
                last_error = exc
                log.warning("Loading PP-OCRv6 failed (attempt %d): %s", attempt + 1, exc)
                time.sleep(2 * (attempt + 1))
        else:
            raise RuntimeError(f"Could not load the OCR models: {last_error}") from last_error
        try:
            self.active_provider = engine.text_det.session.session.get_providers()[0]
        except Exception:
            self.active_provider = "unknown"
        log.info("Fast OCR ready: PP-OCRv6 %s on %s", self.size, self.active_provider)
        return engine

    def engine(self):
        with self._lock:
            if self._engine is None:
                self._engine = self._build()
            return self._engine

    def warm_up(self) -> str:
        """Download/load models and run once. Returns the execution provider in use."""
        image = Image.new("RGB", (480, 120), "white")
        ImageDraw.Draw(image).text((20, 40), "LetterEye warm-up 2026", fill="black", font_size=28)
        self.recognize([image])
        return self.active_provider

    def recognize(self, images: list[Image.Image]) -> OcrResult:
        engine = self.engine()
        start = time.perf_counter()
        page_texts, scores, weights = [], [], []
        for image in images:
            result = engine(image.convert("RGB"))  # RapidOCR converts PIL (RGB) to its BGR format itself
            if result is None or result.txts is None or result.boxes is None:
                page_texts.append("")
                continue
            page_texts.append(_join_lines(result.boxes, result.txts))
            for text, score in zip(result.txts, result.scores, strict=False):
                scores.append(float(score))
                weights.append(max(len(text), 1))
        confidence = float(np.average(scores, weights=weights)) if scores else 0.0
        return OcrResult(
            text="\n\n".join(t for t in page_texts if t).strip(),
            source="fast_ocr",
            confidence=confidence,
            seconds=time.perf_counter() - start,
            device=self.active_provider,
            pages=len(images),
            details={"lines": len(scores), "model": f"PP-OCRv6 {self.size}"},
        )


def _join_lines(boxes, texts) -> str:
    """Group detected text boxes into visual lines (top-to-bottom, left-to-right)."""
    items = []
    for box, text in zip(boxes, texts, strict=False):
        box = np.asarray(box)
        ys, xs = box[:, 1], box[:, 0]
        items.append((float(ys.mean()), float(xs.min()), float(ys.max() - ys.min()), str(text)))
    items.sort(key=lambda item: (item[0], item[1]))
    lines: list[list[tuple[float, float, float, str]]] = []
    for item in items:
        if lines:
            last = lines[-1]
            ref_y = sum(i[0] for i in last) / len(last)
            ref_h = max(i[2] for i in last)
            if abs(item[0] - ref_y) < max(ref_h, item[2]) * 0.5:
                last.append(item)
                continue
        lines.append([item])
    return "\n".join("   ".join(i[3] for i in sorted(line, key=lambda i: i[1])) for line in lines)
