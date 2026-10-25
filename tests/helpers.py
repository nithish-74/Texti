"""Shared test helpers."""

from __future__ import annotations

import tempfile
from pathlib import Path

from texti.config import Config
from texti.index.embedder import HashingEmbedder
from texti.index.store import Store
from texti.secrets_store import MemorySecretStore


class Env:
    """A throwaway data dir, in-memory secrets and the offline hashing embedder."""

    def __init__(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(prefix="m2t-test-")
        self.dir = Path(self._tmp.name)
        self.secrets = MemorySecretStore()
        self.store = Store(self.dir / "data", self.secrets)
        self.embedder = HashingEmbedder()
        self.cfg = Config()

    def write(self, name: str, content: bytes | str) -> Path:
        p = self.dir / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(content if isinstance(content, bytes) else content.encode("utf-8"))
        return p

    def close(self) -> None:
        self.store.close()
        self._tmp.cleanup()


def text_image(lines: list[str], size: tuple[int, int] = (1000, 0)) -> bytes:
    """PNG bytes of black text on white, for OCR tests."""
    import io

    from PIL import Image, ImageDraw, ImageFont

    try:
        font = ImageFont.load_default(size=28)
    except TypeError:  # very old Pillow
        font = ImageFont.load_default()
    height = size[1] or (60 + 56 * len(lines))
    img = Image.new("RGB", (size[0], height), "white")
    draw = ImageDraw.Draw(img)
    for i, line in enumerate(lines):
        draw.text((24, 24 + 56 * i), line, fill="black", font=font)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()
