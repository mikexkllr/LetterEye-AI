from dataclasses import dataclass, field


@dataclass
class OcrResult:
    text: str
    source: str  # text_layer | fast_ocr | ocr_llm
    confidence: float | None = None
    seconds: float = 0.0
    device: str = ""
    pages: int = 0
    details: dict = field(default_factory=dict)
