"""OCR with Tesseract, returning text plus a 0 to 1 confidence."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class Segment:
    """One page or image worth of text."""

    text: str
    offset: int  # page number (1-based) or 0 for a single image
    confidence: float  # 0 to 1; 1.0 for a real PDF text layer
    language: str = ""


class OcrError(RuntimeError):
    pass


def ensure_tessdata_prefix() -> None:
    """Ensure Tesseract uses ~/.texti/tessdata if models are installed there and prefix not set."""
    import os

    if "TESSDATA_PREFIX" not in os.environ:
        from ..config import data_dir

        local_tessdata = data_dir() / "tessdata"
        if local_tessdata.is_dir() and any(local_tessdata.glob("*.traineddata")):
            os.environ["TESSDATA_PREFIX"] = str(local_tessdata)


def installed_languages() -> list[str]:
    import pytesseract

    ensure_tessdata_prefix()
    try:
        return [l for l in pytesseract.get_languages(config="") if l != "osd"]
    except Exception as e:  # noqa: BLE001
        raise OcrError("Tesseract is not installed or not on PATH") from e


def check_languages(lang: str) -> None:
    have = set(installed_languages())
    missing = [l for l in lang.split("+") if l not in have]
    if missing:
        raise OcrError(
            f"Tesseract language data missing for: {', '.join(missing)}. Installed: {', '.join(sorted(have))}. "
            "Install the matching tesseract-ocr-<lang> package (for example tesseract-ocr-tel for Telugu)."
        )


def resolve_ocr_languages(lang: str = "eng") -> str:
    """Combine available language packs if default 'eng' is passed and Indic models are present."""
    if lang not in ("eng", "auto"):
        return lang
    try:
        have = set(installed_languages())
    except Exception:
        return lang
    # If regional Indic languages are installed, include them automatically for higher OCR accuracy
    extras = [l for l in ("hin", "tel") if l in have]
    if extras:
        return "+".join(["eng"] + extras)
    return lang


def ocr_image(image, lang: str = "eng", offset: int = 0) -> Segment:
    """OCR a PIL image or a file path."""
    import pytesseract
    from PIL import Image

    lang = resolve_ocr_languages(lang)
    check_languages(lang)

    opened = False
    if isinstance(image, (str, Path)):
        image = Image.open(image)
        opened = True
    try:
        if image.mode not in ("RGB", "L"):
            image = image.convert("RGB")
        data = pytesseract.image_to_data(image, lang=lang, output_type=pytesseract.Output.DICT)
    finally:
        if opened:
            image.close()

    lines: dict[tuple[int, int, int], list[str]] = {}
    confs: list[float] = []
    for i, word in enumerate(data["text"]):
        word = word.strip()
        conf = float(data["conf"][i])
        if not word or conf < 0:
            continue
        key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
        lines.setdefault(key, []).append(word)
        confs.append(conf)
    text = "\n".join(" ".join(ws) for _, ws in sorted(lines.items()))
    confidence = (sum(confs) / len(confs) / 100.0) if confs else 0.0
    return Segment(text=text, offset=offset, confidence=round(confidence, 3), language=lang)
