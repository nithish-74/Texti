"""Tier decision: folder rules, document-type phrases, general-content cues, patterns.

Design rules:
- A folder rule sets the starting tier. With no rule the starting tier is
  ``default_tier`` (2).
- Content signals can only RAISE the tier, never lower it. A file the user put in a
  "Tier 1" folder still becomes Tier 3 if it contains an Aadhaar number.
- Lecture-style content with no sensitive signal and no folder rule is Tier 1.
  This is a keyword heuristic standing in for the small local classifier planned for
  a later phase; it is deliberately conservative.
- When unsure, the higher tier wins.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .patterns import Finding, detect, required_tier

# A strong phrase for a kind of DOCUMENT makes the whole file Tier 3: every page of a
# bank statement or a prescription is private, even the pages with no number on them.
PRIVATE_PHRASES = (
    "bank statement",
    "account statement",
    "statement of account",
    "passbook",
    "cheque book",
    "income tax return",
    "form 16",
    "prescription",
    "diagnosis",
    "discharge summary",
    "lab report",
    "pathology report",
    "patient name",
    "medical record",
    "last will and testament",
    "non-disclosure agreement",
)

# Phrases that mark an identity document apply to the PAGE they appear on. A scanned ID in
# an otherwise ordinary PDF makes that page Tier 3 without making the whole file private,
# and this catches an ID page whose number OCR garbled so no pattern matched.
PAGE_PRIVATE_PHRASES = (
    "aadhaar",
    "aadhar",
    "unique identification authority",
    "permanent account number",
    "income tax department",
    "passport no",
    "republic of india passport",
    "driving licence",
    "driving license",
    "voter id",
    "election commission of india",
    # Generic wording, for ID cards whose number matches no known pattern.
    "identity card",
    "identification card",
    "id card",
    "national id",
    "id number",
    "id no.",
    "id no:",
    "employee id",
    "student id",
    "staff id",
    "government of india",
    "govt of india",
    "government-issued",
)


# Cues that a document is ordinary study material. Two distinct cues are required.
GENERAL_CUES = (
    "lecture",
    "syllabus",
    "chapter",
    "unit ",
    "module ",
    "slide",
    "theorem",
    "definition",
    "assignment",
    "tutorial",
    "course",
    "semester",
    "learning outcomes",
    "references",
    "textbook",
)


@dataclass(frozen=True)
class PathRule:
    prefix: str  # OneDrive path prefix, compared case-insensitively
    tier: int


@dataclass
class TierDecision:
    tier: int
    reasons: list[str] = field(default_factory=list)
    findings: list[Finding] = field(default_factory=list)


def _norm(path: str) -> str:
    return path.replace("\\", "/").strip("/").lower()


def rule_tier(path: str, rules: list[PathRule]) -> int | None:
    """Tier from the longest matching path rule, or None when no rule matches."""
    p = _norm(path)
    best: PathRule | None = None
    for r in rules:
        prefix = _norm(r.prefix)
        if p == prefix or p.startswith(prefix + "/"):
            if best is None or len(prefix) >= len(_norm(best.prefix)):  # on a tie the later rule wins
                best = r
    return best.tier if best else None


def document_base_tier(
    text: str,
    path: str = "",
    rules: list[PathRule] | None = None,
    default_tier: int = 2,
) -> tuple[int, list[str]]:
    """Tier from rules, phrases and cues only (no pattern findings).

    Pattern findings are applied per page so one ID page does not make a whole lecture
    PDF private. Document-type phrases, however, apply to the whole file.
    """
    reasons: list[str] = []
    ruled = rule_tier(path, rules or [])
    if ruled is not None:
        tier = ruled
        reasons.append(f"folder rule: tier {ruled}")
    else:
        tier = default_tier
        reasons.append(f"no folder rule: default tier {default_tier}")

    lowered = text.lower()
    hits = [p for p in PRIVATE_PHRASES if p in lowered]
    if hits:
        if tier < 3:
            reasons.append(f"private document phrase: {hits[0]!r}")
        tier = 3
    elif ruled is None:
        cues = [c for c in GENERAL_CUES if c in lowered]
        if len(cues) >= 2 and required_tier(detect(text)) <= 1:
            tier = min(tier, 1)
            reasons.append("general study-material cues")
    return tier, reasons


def classify_chunk(chunk_text: str, base_tier: int) -> TierDecision:
    """Raise ``base_tier`` using pattern findings in one piece of text.

    The pipeline calls this once per page or image (a "segment"), on the whole segment
    text before it is chunked. That keeps a number from being cut in half by a chunk
    boundary, so a pattern is never missed because of where a chunk happened to end.
    """
    findings = detect(chunk_text)
    needed = required_tier(findings)
    reasons: list[str] = []
    tier = base_tier
    if needed > tier:
        tier = needed
        kinds = sorted({f.kind for f in findings})
        reasons.append(f"detected: {', '.join(kinds)}")
    lowered = chunk_text.lower()
    if tier < 3 and any(p in lowered for p in PAGE_PRIVATE_PHRASES):
        tier = 3
        reasons.append("identity-document wording on this page")
    return TierDecision(tier=tier, reasons=reasons, findings=findings)


def classify(
    text: str,
    path: str = "",
    rules: list[PathRule] | None = None,
    default_tier: int = 2,
) -> TierDecision:
    """Whole-text decision: base tier raised by any pattern finding anywhere."""
    base, reasons = document_base_tier(text, path, rules, default_tier)
    chunk = classify_chunk(text, base)
    return TierDecision(tier=chunk.tier, reasons=reasons + chunk.reasons, findings=chunk.findings)


_TIER_WORDS = re.compile(r"^[123]$")


def parse_tier(value: str | int | None) -> int | None:
    """Accept 1, 2, 3 (or their string forms) and return an int, else None."""
    if value is None:
        return None
    s = str(value).strip()
    return int(s) if _TIER_WORDS.match(s) else None
