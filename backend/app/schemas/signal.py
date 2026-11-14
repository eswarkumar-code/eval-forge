"""
Failure-signal evidence schema.

This is the enforcement point for the single most important rule in the
product: a detected "signal" is not truth until a human approves it.
`source` must always be present and must always be the honest origin of
the signal. Nothing is allowed to relabel a model_generated signal as
human_approved without an actual review record.
"""
from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field

SignalSource = Literal["deterministic", "heuristic", "model_generated", "human_approved"]

# Fixed vocabulary for V1. Keeping this closed (rather than free-text)
# keeps clustering/dashboard category counts meaningful.
SignalType = Literal[
    "thumbs_down",
    "http_or_api_error",
    "malformed_json_output",
    "retry_occurred",
    "user_correction_present",
    "response_edited",
    "large_divergence_after_retry",
    "latency_outlier",
    "refusal_mismatch_suspected",
    "unsupported_claim_suspected",
    "timeout",
]


class SignalEvidence(BaseModel):
    signal_id: str
    interaction_id: str

    signal_type: SignalType
    source: SignalSource

    score: Optional[float] = None  # 0..1 confidence where meaningful; None for boolean deterministic signals
    evidence: list[str] = Field(default_factory=list)  # short human-readable justifications, never raw unredacted text

    created_at: datetime
