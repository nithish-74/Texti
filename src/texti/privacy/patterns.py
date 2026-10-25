"""Detectors for sensitive data in Indian documents.

Every detector returns Finding objects with character spans. Detectors that have a
checksum (Aadhaar, card numbers) validate it, so random digit runs are not flagged.
Findings never get logged with their text: callers must treat ``Finding.text`` as
sensitive.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# --- checksums ---------------------------------------------------------------

_VERHOEFF_D = (
    (0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
    (1, 2, 3, 4, 0, 6, 7, 8, 9, 5),
    (2, 3, 4, 0, 1, 7, 8, 9, 5, 6),
    (3, 4, 0, 1, 2, 8, 9, 5, 6, 7),
    (4, 0, 1, 2, 3, 9, 5, 6, 7, 8),
    (5, 9, 8, 7, 6, 0, 4, 3, 2, 1),
    (6, 5, 9, 8, 7, 1, 0, 4, 3, 2),
    (7, 6, 5, 9, 8, 2, 1, 0, 4, 3),
    (8, 7, 6, 5, 9, 3, 2, 1, 0, 4),
    (9, 8, 7, 6, 5, 4, 3, 2, 1, 0),
)
_VERHOEFF_P = (
    (0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
    (1, 5, 7, 6, 2, 8, 3, 0, 9, 4),
    (5, 8, 0, 3, 7, 9, 6, 1, 4, 2),
    (8, 9, 1, 6, 0, 4, 3, 5, 2, 7),
    (9, 4, 5, 3, 1, 2, 6, 8, 7, 0),
    (4, 2, 8, 6, 5, 7, 3, 9, 0, 1),
    (2, 7, 9, 3, 8, 0, 6, 4, 1, 5),
    (7, 0, 4, 6, 9, 1, 3, 2, 5, 8),
)
_VERHOEFF_INV = (0, 4, 3, 2, 1, 5, 6, 7, 8, 9)


def verhoeff_valid(number: str) -> bool:
    """True when the digit string passes the Verhoeff checksum."""
    if not number.isdigit():
        return False
    c = 0
    for i, ch in enumerate(reversed(number)):
        c = _VERHOEFF_D[c][_VERHOEFF_P[i % 8][int(ch)]]
    return c == 0


def verhoeff_check_digit(number: str) -> str:
    """Check digit to append to ``number`` so the result passes verhoeff_valid."""
    c = 0
    for i, ch in enumerate(reversed(number)):
        c = _VERHOEFF_D[c][_VERHOEFF_P[(i + 1) % 8][int(ch)]]
    return str(_VERHOEFF_INV[c])


def luhn_valid(number: str) -> bool:
    """True when the digit string passes the Luhn checksum."""
    if not number.isdigit():
        return False
    total = 0
    for i, ch in enumerate(reversed(number)):
        d = int(ch)
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


# --- findings ----------------------------------------------------------------


@dataclass(frozen=True)
class KindInfo:
    """How a kind of finding affects classification and masking."""

    min_tier: int  # the document or chunk is raised to at least this tier
    mask_from_tier: int  # masked when the final tier is >= this


KINDS: dict[str, KindInfo] = {
    "aadhaar": KindInfo(3, 1),
    "pan": KindInfo(3, 1),
    "passport": KindInfo(3, 1),
    "bank_account": KindInfo(3, 1),
    "ifsc": KindInfo(2, 2),
    "card": KindInfo(2, 1),
    "otp": KindInfo(2, 1),
    "secret": KindInfo(2, 1),
    "phone": KindInfo(1, 2),
    "email": KindInfo(1, 2),
}


@dataclass(frozen=True)
class Finding:
    kind: str
    start: int
    end: int
    text: str  # sensitive: never log this

    def __repr__(self) -> str:  # keep the matched text out of logs and tracebacks
        return f"Finding(kind={self.kind!r}, start={self.start}, end={self.end})"


# --- regexes -----------------------------------------------------------------

_AADHAAR = re.compile(r"(?<!\d)([2-9]\d{3}[ -]?\d{4}[ -]?\d{4})(?!\d)")
# 4th letter is the holder type: P person, C company, H HUF, F firm, A AOP, T trust,
# B body of individuals, L local authority, J juridical person, G government.
_PAN = re.compile(r"(?<![A-Za-z0-9])([A-Z]{3}[ABCFGHLJPT][A-Z]\d{4}[A-Z])(?![A-Za-z0-9])")
_PASSPORT = re.compile(r"(?<![A-Za-z0-9])([A-PR-WY][1-9]\d{6})(?![A-Za-z0-9])")
_IFSC = re.compile(r"(?<![A-Za-z0-9])([A-Z]{4}0[A-Z0-9]{6})(?![A-Za-z0-9])")
_CARD = re.compile(r"(?<!\d)(\d(?:[ -]?\d){12,18})(?!\d)")
_BANK_ACCOUNT = re.compile(
    r"(?i)(?:a/c|acct|account)(?:\s*(?:no\.?|number|num|#))?\s*[:\-]?\s*(\d{9,18})(?!\d)"
)
_OTP = re.compile(r"(?i)\b(?:otp|one[- ]time (?:password|passcode)|verification code)\b[^\d\n]{0,25}(\d{4,8})(?!\d)")
_PHONE = re.compile(r"(?<![\d])((?:\+91[ -]?|0)?[6-9]\d{4}[ -]?\d{5})(?![\d])")
_EMAIL = re.compile(r"(?<![\w.+-])([A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+)")
_SECRETS = (
    re.compile(r"(?<![A-Za-z0-9])(AKIA[0-9A-Z]{16})(?![A-Za-z0-9])"),  # AWS access key id
    re.compile(r"(?<![A-Za-z0-9])(sk-[A-Za-z0-9_-]{20,})"),  # common API key shape
    re.compile(r"(?<![A-Za-z0-9])(gh[pousr]_[A-Za-z0-9]{30,})"),  # GitHub tokens
    re.compile(r"(?i)\b(?:password|passwd|pwd|passcode|pin)\b\s*(?:is|[:=])\s*(\S{3,})"),
)


def _digits(s: str) -> str:
    return re.sub(r"\D", "", s)


def is_valid_ifsc(code: str) -> bool:
    """Validate IFSC code structure and reject repetitive OCR artifacts (e.g. OOOO0OOOOOO)."""
    if len(code) != 11 or code[4] != "0":
        return False
    bank = code[:4]
    branch = code[5:]
    # Bank code requires at least 2 distinct letters
    if len(set(bank)) < 2:
        return False
    # Branch code should not be all '0's or all 'O's
    if len(set(branch)) < 2 and branch[0] in ("0", "O"):
        return False
    return True


def detect(text: str) -> list[Finding]:
    """Find sensitive spans. Overlaps are resolved in priority order (most specific first)."""
    findings: list[Finding] = []
    claimed: list[tuple[int, int]] = []

    def free(start: int, end: int) -> bool:
        return all(end <= a or start >= b for a, b in claimed)

    def add(kind: str, m: re.Match[str], group: int = 1) -> None:
        start, end = m.span(group)
        if free(start, end):
            claimed.append((start, end))
            findings.append(Finding(kind, start, end, text[start:end]))

    for m in _AADHAAR.finditer(text):
        if verhoeff_valid(_digits(m.group(1))):
            add("aadhaar", m)
    for m in _PAN.finditer(text):
        add("pan", m)
    for m in _BANK_ACCOUNT.finditer(text):
        add("bank_account", m)
    for m in _CARD.finditer(text):
        digits = _digits(m.group(1))
        if 13 <= len(digits) <= 19 and luhn_valid(digits):
            add("card", m)
    for m in _IFSC.finditer(text):
        if is_valid_ifsc(m.group(1)):
            add("ifsc", m)

    for m in _PASSPORT.finditer(text):
        add("passport", m)
    for m in _OTP.finditer(text):
        add("otp", m)
    for pattern in _SECRETS:
        for m in pattern.finditer(text):
            add("secret", m)
    for m in _EMAIL.finditer(text):
        add("email", m)
    for m in _PHONE.finditer(text):
        add("phone", m)

    findings.sort(key=lambda f: f.start)
    return findings


def required_tier(findings: list[Finding]) -> int:
    """Lowest tier that is safe for these findings (1 when there are none)."""
    return max((KINDS[f.kind].min_tier for f in findings), default=1)
