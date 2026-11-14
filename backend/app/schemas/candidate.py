"""
Candidate evaluation case schema.

CRITICAL RULE (do not violate): `proposed_expected_output` and
`proposed_failure_category` are never automatically promoted into the
canonical evaluation dataset. Only `reviewer_expected_output` /
`reviewer_failure_category`, set via an actual human review action with
review_status == "approved", may enter the versioned dataset. Enforcement
lives in backend/app/dataset/, not just here — but the schema keeps the
proposed_* and reviewer_* fields separate so it's structurally impossible
to confuse them.
"""
from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field

GenerationSource = Literal["user_correction", "deterministic", "heuristic", "model_generated"]
Difficulty = Literal["easy", "medium", "hard"]
Severity = Literal["low", "medium", "high", "critical"]
ReviewStatus = Literal["pending", "approved", "rejected"]


class EvalCandidate(BaseModel):
    candidate_id: str

    source_interaction_ids: list[str] = Field(default_factory=list)

    input_text: str
    context: Optional[str] = None

    observed_output: Optional[str] = None

    proposed_expected_output: Optional[str] = None
    proposed_failure_category: Optional[str] = None

    generation_source: GenerationSource

    difficulty: Difficulty = "medium"
    severity: Severity = "medium"

    cluster_id: Optional[str] = None

    review_status: ReviewStatus = "pending"

    reviewer_expected_output: Optional[str] = None
    reviewer_failure_category: Optional[str] = None
    reviewer_notes: Optional[str] = None

    created_at: datetime
    reviewed_at: Optional[datetime] = None

    def is_canonical_ready(self) -> bool:
        """A candidate may enter the versioned dataset only if a human
        actually approved it AND left a reviewer-sourced expected output
        and category (not just a rubber-stamp of the proposed_* fields
        with no reviewer_* set). Checks for blank/whitespace-only strings,
        not just None -- found necessary via Gate 1 live acceptance
        testing, where a real model returned an empty string rather than
        null for an ungroundable field."""
        return (
            self.review_status == "approved"
            and bool(self.reviewer_expected_output)
            and bool(self.reviewer_expected_output.strip())
            and bool(self.reviewer_failure_category)
            and bool(self.reviewer_failure_category.strip())
        )
