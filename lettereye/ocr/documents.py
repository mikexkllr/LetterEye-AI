"""Reading input documents (PDF and images) without keeping file handles open.

Files are read into memory first: on Windows an open handle would stop us from moving the file later.
"""

from __future__ import annotations

import io
from pathlib import Path

import pypdfium2 as pdfium
from PIL import Image, ImageOps

PDF_EXTENSIONS = {".pdf"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
SUPPORTED_EXTENSIONS = PDF_EXTENSIONS | IMAGE_EXTENSIONS


def is_supported(path: str | Path) -> bool:
    return Path(path).suffix.lower() in SUPPORTED_EXTENSIONS


class LoadedDocument:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.data = self.path.read_bytes()
        self.is_pdf = self.path.suffix.lower() in PDF_EXTENSIONS or self.data[:5] == b"%PDF-"

    def page_count(self) -> int:
        if self.is_pdf:
            pdf = pdfium.PdfDocument(self.data)
            try:
                return len(pdf)
            finally:
                pdf.close()
        with Image.open(io.BytesIO(self.data)) as img:
            return getattr(img, "n_frames", 1)

    def text_layer(self, max_pages: int) -> str:
        """Embedded text of a digital PDF (empty for scans and images)."""
        if not self.is_pdf:
            return ""
        pdf = pdfium.PdfDocument(self.data)
        try:
            texts = []
            for index in range(min(len(pdf), max_pages)):
                page = pdf[index]
                textpage = page.get_textpage()
                texts.append(textpage.get_text_bounded())
                textpage.close()
                page.close()
            return "\n".join(texts).replace("\r\n", "\n").strip()
        finally:
            pdf.close()

    def render(self, max_pages: int, dpi: int = 200) -> list[Image.Image]:
        """Page images (RGB) for OCR."""
        if self.is_pdf:
            pdf = pdfium.PdfDocument(self.data)
            try:
                images = []
                for index in range(min(len(pdf), max_pages)):
                    page = pdf[index]
                    images.append(page.render(scale=dpi / 72).to_pil().convert("RGB"))
                    page.close()
                return images
            finally:
                pdf.close()
        images = []
        with Image.open(io.BytesIO(self.data)) as img:
            for frame in range(min(getattr(img, "n_frames", 1), max_pages)):
                img.seek(frame)
                images.append(ImageOps.exif_transpose(img.copy()).convert("RGB"))
        return images


def save_preview(image: Image.Image, target: Path, max_side: int = 1100) -> Path:
    preview = image.copy()
    preview.thumbnail((max_side, max_side))
    target.parent.mkdir(parents=True, exist_ok=True)
    preview.save(target, format="PNG", optimize=True)
    return target
