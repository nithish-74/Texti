"""Turn a local file into text segments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .audio import AUDIO_EXTENSIONS, MEDIA_AUDIO_EXTENSIONS, VIDEO_EXTENSIONS, extract_audio
from .ocr import Segment, ocr_image
from .pdf import extract_pdf

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tif", ".tiff", ".webp"}
SUPPORTED_EXTENSIONS = IMAGE_EXTENSIONS | {".pdf"} | MEDIA_AUDIO_EXTENSIONS


@dataclass
class ExtractionResult:
    segments: list[Segment]
    pages: int


def is_supported(name: str) -> bool:
    return Path(name).suffix.lower() in SUPPORTED_EXTENSIONS


def extract_file(path: Path, language: str = "eng") -> ExtractionResult:
    ext = Path(path).suffix.lower()
    if ext == ".pdf":
        segments, pages = extract_pdf(path, lang=language)
        return ExtractionResult(segments, pages)
    if ext in IMAGE_EXTENSIONS:
        seg = ocr_image(path, lang=language)
        return ExtractionResult([seg] if seg.text.strip() else [], 1)
    if ext in MEDIA_AUDIO_EXTENSIONS:
        segments, pages = extract_audio(path, language=language)
        return ExtractionResult(segments, pages)
    raise ValueError(f"unsupported file type: {ext or '(none)'}")

