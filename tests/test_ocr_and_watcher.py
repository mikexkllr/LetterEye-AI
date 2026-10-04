import time
from pathlib import Path

from lettereye.ocr.documents import LoadedDocument, is_supported
from lettereye.ocr.fast_ocr import _join_lines
from lettereye.ocr.llm_ocr import clean_ocr_output, prompt_for
from lettereye.services.watcher import is_candidate, wait_until_ready
from lettereye.settings import Settings, SettingsStore


def test_supported_files_and_temp_files():
    assert is_supported("a.PDF") and is_supported("b.tiff") and not is_supported("c.docx")
    assert is_candidate(Path("scan.pdf"))
    assert not is_candidate(Path("~$scan.pdf"))
    assert not is_candidate(Path("scan.pdf.part"))
    assert not is_candidate(Path(".hidden.pdf"))


def test_wait_until_ready(tmp_path):
    path = tmp_path / "scan.pdf"
    path.write_bytes(b"%PDF-1.4 complete")
    started = time.monotonic()
    assert wait_until_ready(path, interval=0.1)
    assert time.monotonic() - started < 3
    assert not wait_until_ready(tmp_path / "missing.pdf", interval=0.1)


def test_image_document_rendering(scan_image):
    doc = LoadedDocument(scan_image)
    assert not doc.is_pdf and doc.page_count() == 1 and doc.text_layer(2) == ""
    assert doc.render(2)[0].size == (400, 560)


def test_line_grouping_keeps_reading_order():
    boxes = [
        [[300, 10], [400, 10], [400, 30], [300, 30]],   # right part of line 1
        [[10, 12], [100, 12], [100, 32], [10, 32]],     # left part of line 1
        [[10, 60], [200, 60], [200, 80], [10, 80]],     # line 2
    ]
    assert _join_lines(boxes, ["Datum", "Herrn", "Bob Smith"]) == "Herrn   Datum\nBob Smith"


def test_llm_ocr_cleanup_removes_fences_and_repeats():
    raw = "```markdown\nLine one\nLine two\nLine three\nLine four\nLine one\nLine two\nLine three\nLine four\n```"
    assert clean_ocr_output(raw) == "Line one\nLine two\nLine three\nLine four"
    assert prompt_for("glm-ocr:q8_0") == "Text Recognition:"
    assert prompt_for("deepseek-ocr") == "Free OCR."
    assert "Transcribe" in prompt_for("qwen3.5:9b")


def test_settings_roundtrip_and_problems(tmp_path):
    store = SettingsStore(tmp_path / "settings.json")
    assert not store.get().setup_completed
    store.update(language="en", inbox_folder=str(tmp_path), output_folder=str(tmp_path))
    reloaded = SettingsStore(tmp_path / "settings.json").get()
    assert reloaded.language == "en"
    assert any("different" in p for p in reloaded.problems())
    (tmp_path / "settings.json").write_text("{ broken", encoding="utf-8")
    assert SettingsStore(tmp_path / "settings.json").get() == Settings()


def test_llm_ocr_stops_streaming_when_the_page_repeats():
    from langchain_core.messages import AIMessageChunk
    from PIL import Image

    from lettereye.ocr.llm_ocr import LlmOCR

    page = ["Stadtwerke München GmbH\n", "Herrn\n", "Bob Smith\n", "Sehr geehrter Herr Smith,\n"]

    class LoopingModel:
        def __init__(self):
            self.chunks_sent = 0

        def stream(self, messages):
            for chunk in page + ["```markdown\n"] + page * 50:  # starts over and would never stop
                self.chunks_sent += 1
                yield AIMessageChunk(content=chunk)

    model = LoopingModel()
    result = LlmOCR("http://unused", "glm-ocr", llm=model).recognize([Image.new("RGB", (100, 100), "white")])
    assert result.text == "".join(page).strip()
    assert model.chunks_sent < 10  # stopped right after the repeat began
