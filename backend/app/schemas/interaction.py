"""
Canonical normalized production interaction log schema.

Every raw interaction MUST be converted into this schema before anything
else in the pipeline runs (dedup, signal extraction, clustering, candidate
generation, regression). Nothing downstream may operate on raw/unredacted
input — see backend/app/redaction/ for the enforcement point.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, field_validator

Status = Literal["success", "error", "timeout", "malformed_output", "refusal"]
UserFeedback = Literal["positive", "negative", "neutral", "none"]
Provenance = Literal["raw_log", "synthetic_demo"]


class InteractionLog(BaseModel):
    """The single canonical unit of data in the system.

    Design rules (do not violate silently — see docs/design_lock.md):
      * A retry is NOT a correction. Retry = another model attempt at
        substantially the same request (system-initiated or user-initiated
        re-ask). Correction = a human explicitly supplies/edits what the
        response should have been.
      * Multi-turn conversations are never collapsed into one string.
        conversation_id + interaction_id + parent_interaction_id preserve
        sequence while keeping each turn independently analyzable.
      * Every object downstream of this one must carry provenance back to
        the original interaction_id(s) that produced it.
      * pii_detected / pii_types describe the ORIGINAL content. By the time
        a record reaches persistence, user_input/system_prompt/
        assistant_output/user_correction/edited_response must already be
        redacted (see redaction policy) — this schema does not itself
        redact, it only records that redaction happened and what was found.
    """

    model_config = {"protected_namespaces": ()}


    interaction_id: str
    conversation_id: Optional[str] = None
    parent_interaction_id: Optional[str] = None

    timestamp: datetime
    application: str
    feature: Optional[str] = None

    model_provider: str
    model_name: str

    system_prompt: Optional[str] = None
    user_input: str
    assistant_output: Optional[str] = None

    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    latency_ms: Optional[float] = None

    status: Status
    user_feedback: UserFeedback = "none"

    retry_of: Optional[str] = None
    retry_count: int = 0

    user_correction: Optional[str] = None
    edited_response: Optional[str] = None

    error_type: Optional[str] = None
    error_message_redacted: Optional[str] = None

    structured_output_valid: Optional[bool] = None

    pii_detected: bool = False
    pii_types: list[str] = Field(default_factory=list)

    provenance: Provenance

    # content hash used for exact-duplicate detection (populated by
    # ingestion, not supplied by the caller)
    content_hash: Optional[str] = None

    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("retry_count")
    @classmethod
    def _retry_count_non_negative(cls, v: int) -> int:
        if v < 0:
            raise ValueError("retry_count must be >= 0")
        return v

    @field_validator("retry_of")
    @classmethod
    def _retry_of_not_self(cls, v: Optional[str], info) -> Optional[str]:
        iid = info.data.get("interaction_id")
        if v is not None and iid is not None and v == iid:
            raise ValueError("retry_of cannot reference its own interaction_id")
        return v


class RawIngestRecord(BaseModel):
    """What we accept on the wire before validation/redaction.

    Deliberately permissive (Any-ish) — malformed input must be able to
    reach this model so it can be quarantined rather than crashing ingestion.
    """

    model_config = {"extra": "allow", "protected_namespaces": ()}


    interaction_id: Optional[str] = None
    conversation_id: Optional[str] = None
    parent_interaction_id: Optional[str] = None
    timestamp: Optional[str] = None
    application: Optional[str] = None
    feature: Optional[str] = None
    model_provider: Optional[str] = None
    model_name: Optional[str] = None
    system_prompt: Optional[str] = None
    user_input: Optional[str] = None
    assistant_output: Optional[str] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    latency_ms: Optional[float] = None
    status: Optional[str] = None
    user_feedback: Optional[str] = None
    retry_of: Optional[str] = None
    retry_count: Optional[int] = None
    user_correction: Optional[str] = None
    edited_response: Optional[str] = None
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    structured_output_valid: Optional[bool] = None
    provenance: Optional[str] = None
    metadata: Optional[dict[str, Any]] = None

