"""
Sensitive-data detection and redaction.

LOCKED POLICY (see docs/design_lock.md section 2):
  - Sensitive data is detected and redacted during ingestion, BEFORE
    persistence, clustering, candidate generation, model analysis, or
    export.
  - The unredacted original is never stored by this application.
  - Detection is deterministic/regex-based for V1. We do not claim
    comprehensive PII coverage.
  - Redaction uses stable typed placeholders (<EMAIL_1>, <PHONE_1>, ...).
    Repeated occurrences of the same raw value within one interaction map
    to the same placeholder, to preserve structure useful for clustering.
  - Records that look like they contain credentials/tokens/private keys or
    an unusually dense concentration of sensitive identifiers are
    quarantined instead of being auto-redacted and processed.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

# --- Detection patterns -----------------------------------------------
# Ordered: more specific / higher-risk patterns first, since a single
# span should only be claimed by one pattern type.

_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("api_key_or_token", re.compile(
        r"(?:sk-[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16}|ghp_[A-Za-z0-9]{30,}|"
        r"eyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,})"
    )),
    ("private_key_block", re.compile(
        r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----",
        re.DOTALL,
    )),
    ("ssn", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    ("credit_card", re.compile(r"\b(?:\d[ -]?){13,16}\b")),
    ("email", re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")),
    ("phone", re.compile(
        r"(?<!\d)(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}(?!\d)"
    )),
    ("ip_address", re.compile(
        r"\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d?\d)\b"
    )),
    ("url_with_credentials", re.compile(r"https?://[^\s:/@]+:[^\s:/@]+@[^\s]+")),
]

# Signal-only patterns used purely to decide "quarantine", never redacted
# in place because we don't want to process them at all.
_QUARANTINE_TRIGGERS: list[re.Pattern] = [
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    re.compile(r"(?:sk-[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16}|ghp_[A-Za-z0-9]{30,})"),
]

_DENSE_PII_QUARANTINE_THRESHOLD = 5  # >= this many distinct PII hits in one record -> quarantine


@dataclass
class RedactionResult:
    redacted_text: str
    pii_types: list[str] = field(default_factory=list)
    hit_count: int = 0


def _redact_text(text: str, placeholder_map: dict[str, str], counters: dict[str, int]) -> RedactionResult:
    if not text:
        return RedactionResult(redacted_text=text or "", pii_types=[], hit_count=0)

    found_types: set[str] = set()
    hit_count = 0
    out = text

    for pii_type, pattern in _PATTERNS:
        def _sub(match: re.Match, pii_type=pii_type) -> str:
            nonlocal hit_count
            raw = match.group(0)
            key = f"{pii_type}:{raw}"
            if key not in placeholder_map:
                counters[pii_type] = counters.get(pii_type, 0) + 1
                placeholder_map[key] = f"<{pii_type.upper()}_{counters[pii_type]}>"
            found_types.add(pii_type)
            hit_count += 1
            return placeholder_map[key]

        out = pattern.sub(_sub, out)

    return RedactionResult(redacted_text=out, pii_types=sorted(found_types), hit_count=hit_count)


@dataclass
class InteractionRedactionOutcome:
    quarantine: bool
    quarantine_reason: Optional[str]
    pii_detected: bool
    pii_types: list[str]
    redacted_fields: dict[str, str]


def check_quarantine(raw_fields: dict[str, str]) -> Optional[str]:
    """Return a reason string if the record should be quarantined instead
    of auto-redacted, else None."""
    joined = "\n".join(v for v in raw_fields.values() if v)
    for trigger in _QUARANTINE_TRIGGERS:
        if trigger.search(joined):
            return "credential_or_private_key_detected"
    return None


def redact_interaction_fields(raw_fields: dict[str, Optional[str]]) -> InteractionRedactionOutcome:
    """Redact a set of text fields belonging to ONE interaction, sharing a
    single placeholder map so the same raw value gets the same placeholder
    across system_prompt/user_input/assistant_output/etc.

    raw_fields: e.g. {"system_prompt": ..., "user_input": ..., ...}
    """
    non_empty = {k: v for k, v in raw_fields.items() if v}

    reason = check_quarantine(non_empty)

    placeholder_map: dict[str, str] = {}
    counters: dict[str, int] = {}
    redacted_fields: dict[str, str] = {}
    all_types: set[str] = set()
    total_hits = 0

    for field_name, text in raw_fields.items():
        if text is None:
            continue
        result = _redact_text(text, placeholder_map, counters)
        redacted_fields[field_name] = result.redacted_text
        all_types.update(result.pii_types)
        total_hits += result.hit_count

    if reason is None and total_hits >= _DENSE_PII_QUARANTINE_THRESHOLD:
        reason = "dense_pii_concentration"

    return InteractionRedactionOutcome(
        quarantine=reason is not None,
        quarantine_reason=reason,
        pii_detected=len(all_types) > 0,
        pii_types=sorted(all_types),
        redacted_fields=redacted_fields,
    )
