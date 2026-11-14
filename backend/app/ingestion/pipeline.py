"""
Ingestion pipeline.

Locked order of operations (design_lock.md section 2):
  Raw input -> validate -> sensitive-data scan -> redact -> persist
  normalized record -> dedup -> failure mining -> clustering -> candidate
  generation.

This module implements the first four steps and produces one of two
outcomes per raw record:
  - a redacted, valid InteractionLog ready for persistence, or
  - a QuarantinedRecord (malformed OR flagged sensitive) that is NOT
    processed further automatically.

Deduplication is a separate step (see app/ingestion/dedup.py) that runs
AFTER this pipeline, over the set of successfully normalized records.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional
from uuid import uuid4

from pydantic import ValidationError

from app.redaction.redactor import redact_interaction_fields
from app.schemas.interaction import InteractionLog, RawIngestRecord

REDACTABLE_TEXT_FIELDS = [
    "system_prompt",
    "user_input",
    "assistant_output",
    "user_correction",
    "edited_response",
]

_VALID_STATUS = {"success", "error", "timeout", "malformed_output", "refusal"}
_VALID_FEEDBACK = {"positive", "negative", "neutral", "none"}


@dataclass
class QuarantinedRecord:
    quarantine_id: str
    reason: str
    # Deliberately store only enough to explain WHY it was quarantined —
    # never the raw sensitive content itself, and never persist a full
    # unredacted payload.
    original_interaction_id: Optional[str]
    application: Optional[str]
    detected_at: datetime
    error_detail: str


@dataclass
class IngestOutcome:
    accepted: list[InteractionLog] = field(default_factory=list)
    quarantined: list[QuarantinedRecord] = field(default_factory=list)


def _content_hash(rec: RawIngestRecord) -> str:
    """Hash over normalized-ish content used for exact-duplicate detection
    downstream. Computed on RAW (pre-redaction) values so that redaction
    placeholder numbering (which can vary run to run for a given batch
    ordering) never affects duplicate detection."""
    basis = json.dumps(
        {
            "application": rec.application,
            "model_name": rec.model_name,
            "system_prompt": rec.system_prompt,
            "user_input": rec.user_input,
            "assistant_output": rec.assistant_output,
        },
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(basis.encode("utf-8")).hexdigest()


def _parse_raw(payload: dict[str, Any]) -> tuple[Optional[RawIngestRecord], Optional[str]]:
    try:
        return RawIngestRecord(**payload), None
    except ValidationError as e:
        return None, str(e)


def _coerce_timestamp(raw: Optional[str]) -> Optional[datetime]:
    if not raw:
        return None
    try:
        ts = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        return ts
    except (ValueError, AttributeError):
        return None


def ingest_batch(raw_payloads: list[dict[str, Any]], provenance: str = "raw_log") -> IngestOutcome:
    """Process a batch of raw dict payloads (already parsed from JSON/JSONL)
    through validate -> sensitive-data scan -> redact -> build InteractionLog.

    Malformed records (schema violations, unparseable timestamps, missing
    required fields) and flagged-sensitive records are quarantined rather
    than raising — one bad record must never abort a batch.
    """
    outcome = IngestOutcome()
    now = datetime.now(timezone.utc)

    for idx, payload in enumerate(raw_payloads):
        raw, parse_error = _parse_raw(payload if isinstance(payload, dict) else {})
        if raw is None:
            outcome.quarantined.append(
                QuarantinedRecord(
                    quarantine_id=str(uuid4()),
                    reason="schema_validation_failed",
                    original_interaction_id=(payload or {}).get("interaction_id") if isinstance(payload, dict) else None,
                    application=(payload or {}).get("application") if isinstance(payload, dict) else None,
                    detected_at=now,
                    error_detail=parse_error or "unparseable payload",
                )
            )
            continue

        missing = []
        if not raw.user_input:
            missing.append("user_input")
        if not raw.application:
            missing.append("application")
        if not raw.model_provider:
            missing.append("model_provider")
        if not raw.model_name:
            missing.append("model_name")

        timestamp = _coerce_timestamp(raw.timestamp) if raw.timestamp else now
        if raw.timestamp and timestamp is None:
            missing.append("timestamp (unparseable)")

        status = raw.status if raw.status in _VALID_STATUS else None
        if raw.status and status is None:
            missing.append(f"status (invalid value: {raw.status!r})")
        elif status is None:
            status = "success"

        if missing:
            outcome.quarantined.append(
                QuarantinedRecord(
                    quarantine_id=str(uuid4()),
                    reason="missing_or_invalid_required_fields",
                    original_interaction_id=raw.interaction_id,
                    application=raw.application,
                    detected_at=now,
                    error_detail=f"missing/invalid: {', '.join(missing)}",
                )
            )
            continue

        redaction_fields = {f: getattr(raw, f) for f in REDACTABLE_TEXT_FIELDS}
        redaction = redact_interaction_fields(redaction_fields)

        if redaction.quarantine:
            outcome.quarantined.append(
                QuarantinedRecord(
                    quarantine_id=str(uuid4()),
                    reason=f"sensitive_data:{redaction.quarantine_reason}",
                    original_interaction_id=raw.interaction_id,
                    application=raw.application,
                    detected_at=now,
                    error_detail="record withheld from processing due to sensitive content policy",
                )
            )
            continue

        content_hash = _content_hash(raw)

        try:
            log = InteractionLog(
                interaction_id=raw.interaction_id or str(uuid4()),
                conversation_id=raw.conversation_id,
                parent_interaction_id=raw.parent_interaction_id,
                timestamp=timestamp,
                application=raw.application,
                feature=raw.feature,
                model_provider=raw.model_provider,
                model_name=raw.model_name,
                system_prompt=redaction.redacted_fields.get("system_prompt"),
                user_input=redaction.redacted_fields.get("user_input", raw.user_input),
                assistant_output=redaction.redacted_fields.get("assistant_output"),
                input_tokens=raw.input_tokens,
                output_tokens=raw.output_tokens,
                latency_ms=raw.latency_ms,
                status=status,
                user_feedback=raw.user_feedback if raw.user_feedback in _VALID_FEEDBACK else "none",
                retry_of=raw.retry_of,
                retry_count=raw.retry_count or 0,
                user_correction=redaction.redacted_fields.get("user_correction"),
                edited_response=redaction.redacted_fields.get("edited_response"),
                error_type=raw.error_type,
                error_message_redacted=raw.error_message,  # already scoped to error text; not treated as free text PII surface in V1
                structured_output_valid=raw.structured_output_valid,
                pii_detected=redaction.pii_detected,
                pii_types=redaction.pii_types,
                provenance=provenance,  # type: ignore[arg-type]
                content_hash=content_hash,
                metadata=raw.metadata or {},
            )
        except ValidationError as e:
            outcome.quarantined.append(
                QuarantinedRecord(
                    quarantine_id=str(uuid4()),
                    reason="post_redaction_schema_validation_failed",
                    original_interaction_id=raw.interaction_id,
                    application=raw.application,
                    detected_at=now,
                    error_detail=str(e),
                )
            )
            continue

        outcome.accepted.append(log)

    return outcome
