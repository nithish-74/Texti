"""Masking of sensitive spans before anything is chunked, embedded or stored."""

from __future__ import annotations

from .patterns import KINDS, Finding, detect


def _mask_digits_keep_last4(raw: str, groups: str) -> str:
    digits = [c for c in raw if c.isdigit()]
    if len(digits) <= 4:
        return "X" * len(digits)
    last4 = "".join(digits[-4:])
    if groups == "aadhaar":
        return f"XXXX XXXX {last4}"
    if groups == "card":
        return f"XXXX XXXX XXXX {last4}"
    return "X" * (len(digits) - 4) + last4


def mask_for(finding: Finding) -> str:
    """Replacement text for one finding. Keeps only the last four digits where useful."""
    if finding.kind == "aadhaar":
        return _mask_digits_keep_last4(finding.text, "aadhaar")
    if finding.kind == "card":
        return _mask_digits_keep_last4(finding.text, "card")
    if finding.kind == "bank_account":
        return _mask_digits_keep_last4(finding.text, "plain")
    return f"[REDACTED:{finding.kind.upper()}]"


def redact(text: str, tier: int, findings: list[Finding] | None = None) -> str:
    """Mask every finding whose mask threshold is at or below ``tier``.

    Findings are applied right to left so earlier spans keep their positions.
    """
    if findings is None:
        findings = detect(text)
    out = text
    for f in sorted(findings, key=lambda f: f.start, reverse=True):
        if tier >= KINDS[f.kind].mask_from_tier:
            out = out[: f.start] + mask_for(f) + out[f.end :]
    return out
