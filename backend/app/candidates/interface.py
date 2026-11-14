"""
Candidate generation interface.

Any generator — rule-based or model-backed — produces EvalCandidate
objects with generation_source set honestly to reflect where the
proposed_* fields actually came from. NO generator implementation may set
review_status to anything but "pending", and NO generator may populate
reviewer_expected_output / reviewer_failure_category — those are exclusively
written by the human review module (app/review/).

This interface exists so the pipeline can swap between:
  - DeterministicCandidateGenerator: rule-based, offline, used in the main
    automated test suite (no network/API dependency).
  - ClaudeCandidateGenerator: calls the Anthropic API for a genuinely
    model-assisted proposed category + expected output. Used in production
    and exercised only by a separate, opt-in live acceptance suite.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from app.schemas.candidate import EvalCandidate
from app.schemas.interaction import InteractionLog
from app.schemas.signal import SignalEvidence


@dataclass
class ClusterContext:
    """What a generator gets to see for one cluster it's generating
    candidates for. representative_logs should already be redacted
    InteractionLog objects (redaction happened at ingestion, long before
    this point) — no generator implementation should ever receive raw
    unredacted text."""
    cluster_id: str
    representative_logs: list[InteractionLog]
    signals_by_interaction: dict[str, list[SignalEvidence]]
    dominant_signal_types: list[str]


class CandidateGenerator(ABC):
    """Implementations must NEVER set review_status other than 'pending'
    and must NEVER populate reviewer_* fields."""

    @abstractmethod
    def generate(self, context: ClusterContext) -> list[EvalCandidate]:
        ...

    def _assert_generator_contract(self, candidates: list[EvalCandidate]) -> None:
        for c in candidates:
            assert c.review_status == "pending", "Generators must never set review_status != pending"
            assert c.reviewer_expected_output is None, "Generators must never populate reviewer_expected_output"
            assert c.reviewer_failure_category is None, "Generators must never populate reviewer_failure_category"
            assert c.reviewed_at is None, "Generators must never populate reviewed_at"
