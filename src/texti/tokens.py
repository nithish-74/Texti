"""Token counting.

``tiktoken`` downloads its vocabulary file from the internet the first time it is used,
which would break the rule that the only network calls go to Microsoft Graph. So the
default is a local estimate, and tiktoken is used only when the user opts in with
``use_tiktoken`` in the config (and has already cached the vocabulary).

The estimate is deliberately simple: about 4 ASCII characters per token, and one token
per non-ASCII character (Telugu, Hindi and other Indic scripts tokenize poorly in common
vocabularies, so this errs on the high side). It is an estimate, and every report says so.
"""

from __future__ import annotations

_encoder = None
_use_tiktoken = False


def configure(use_tiktoken: bool) -> None:
    global _use_tiktoken, _encoder
    _use_tiktoken = use_tiktoken
    _encoder = None


def _tiktoken_encoder():
    global _encoder
    if _encoder is None:
        import tiktoken

        _encoder = tiktoken.get_encoding("cl100k_base")
    return _encoder


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    if _use_tiktoken:
        return len(_tiktoken_encoder().encode(text))
    ascii_chars = sum(1 for c in text if ord(c) < 128)
    other = len(text) - ascii_chars
    return max(1, round(ascii_chars / 4) + other)
