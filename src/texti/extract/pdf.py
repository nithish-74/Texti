"""PDF extraction: the text layer first, OCR only for pages that have none."""

from __future__ import annotations

import io
from typing import Callable, Iterable, Protocol

from .ocr import Segment, ocr_image


class PdfPage(Protocol):
    def get_text(self) -> str: ...
    def render_image(self): ...  # returns a PIL image; only called for scanned pages


def extract_pages(
    pages: Iterable[PdfPage],
    lang: str = "eng",
    ocr_fn: Callable = ocr_image,
    min_chars: int = 20,
) -> tuple[list[Segment], int]:
    """Return (segments, page_count). A page with fewer than ``min_chars`` of text is
    treated as scanned and sent to OCR."""
    segments: list[Segment] = []
    count = 0
    for i, page in enumerate(pages, start=1):
        count += 1
        text = (page.get_text() or "").strip()
        if len(text) >= min_chars:
            segments.append(Segment(text=text, offset=i, confidence=1.0, language=lang))
        else:
            seg = ocr_fn(page.render_image(), lang=lang, offset=i)
            if seg.text.strip():
                segments.append(seg)
    return segments, count


class _FitzPage:
    """Adapter from a PyMuPDF page to the PdfPage protocol."""

    def __init__(self, page, dpi: int) -> None:
        self._page = page
        self._dpi = dpi

    def get_text(self) -> str:
        return self._page.get_text("text")

    def render_image(self):
        from PIL import Image

        png = self._page.get_pixmap(dpi=self._dpi).tobytes("png")
        return Image.open(io.BytesIO(png))


def extract_pdf(path, lang: str = "eng", dpi: int = 200, ocr_fn: Callable = ocr_image) -> tuple[list[Segment], int]:
    try:
        import fitz  # PyMuPDF
    except ImportError as e:
        raise RuntimeError("PyMuPDF is not installed: pip install PyMuPDF") from e
    doc = fitz.open(path)
    try:
        return extract_pages((_FitzPage(p, dpi) for p in doc), lang=lang, ocr_fn=ocr_fn)
    finally:
        doc.close()
