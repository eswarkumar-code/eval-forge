"""
Human review service.

This is the ONLY place in the system allowed to set review_status to
"approved" or "rejected", and the ONLY place allowed to populate
reviewer_expected_output / reviewer_failure_category / reviewer_notes /
reviewed_at. No generator, no clustering step, no API layer may bypass
this module to write those fields directly — see
EvalCandidate.is_canonical_ready(), which requires reviewer_* fields to
be human-set, not just proposed_* fields copied over.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from app.schemas.candidate import EvalCandidate


class ReviewError(ValueError):
    pass


@dataclass
class ReviewAction:
    candidate_id: str
    action: str  # "approve" | "reject" | "edit"
    reviewer_expected_output: Optional[str] = None
    reviewer_failure_category: Optional[str] = None
    reviewer_notes: Optional[str] = None


def approve_candidate(
    candidate: EvalCandidate,
    reviewer_expected_output: Optional[str] = None,
    reviewer_failure_category: Optional[str] = None,
    reviewer_notes: Optional[str] = None,
) -> EvalCandidate:
    """Approve a candidate. A reviewer may accept the model/heuristic's
    proposed_* fields as-is by passing them through explicitly here (this
    is a deliberate, visible human action — NOT an automatic promotion),
    or supply edited values. Either way this function is the only path
    that can produce is_canonical_ready() == True.
    """
    final_expected = reviewer_expected_output if reviewer_expected_output is not None else candidate.proposed_expected_output
    final_category = reviewer_failure_category if reviewer_failure_category is not None else candidate.proposed_failure_category

    # Defense in depth: reject blank/whitespace-only values, not just None.
    # Found via Gate 1 live acceptance testing — live Claude output returned
    # an empty string ("") rather than JSON null for an ungroundable case,
    # which an `is None` check alone would have let through, producing a
    # canonical case with a blank expected output. Generators are expected
    # to normalize their own output (see claude_generator._normalize_optional_str),
    # but the gate itself must not depend on every generator doing so correctly.
    if not final_expected or not final_expected.strip():
        raise ReviewError("Cannot approve a candidate with no expected output (neither reviewer-supplied nor proposed).")
    if not final_category or not final_category.strip():
        raise ReviewError("Cannot approve a candidate with no failure category (neither reviewer-supplied nor proposed).")

    return candidate.model_copy(update={
        "review_status": "approved",
        "reviewer_expected_output": final_expected,
        "reviewer_failure_category": final_category,
        "reviewer_notes": reviewer_notes,
        "reviewed_at": datetime.now(timezone.utc),
    })


def reject_candidate(candidate: EvalCandidate, reviewer_notes: Optional[str] = None) -> EvalCandidate:
    return candidate.model_copy(update={
        "review_status": "rejected",
        "reviewer_notes": reviewer_notes,
        "reviewed_at": datetime.now(timezone.utc),
    })


def edit_and_approve(
    candidate: EvalCandidate,
    reviewer_expected_output: str,
    reviewer_failure_category: str,
    reviewer_notes: Optional[str] = None,
) -> EvalCandidate:
    """Explicit edit path: reviewer supplies both fields themselves. Kept
    as a distinct function (rather than just calling approve_candidate
    with kwargs) so API/UI code paths are unambiguous about intent."""
    if not reviewer_expected_output or not reviewer_failure_category:
        raise ReviewError("edit_and_approve requires both reviewer_expected_output and reviewer_failure_category.")
    return approve_candidate(
        candidate,
        reviewer_expected_output=reviewer_expected_output,
        reviewer_failure_category=reviewer_failure_category,
        reviewer_notes=reviewer_notes,
    )


def reset_to_pending(candidate: EvalCandidate) -> EvalCandidate:
    """Allow a reviewer to undo a mistaken approve/reject. Clears
    reviewer_* fields entirely rather than leaving stale values behind."""
    return candidate.model_copy(update={
        "review_status": "pending",
        "reviewer_expected_output": None,
        "reviewer_failure_category": None,
        "reviewer_notes": None,
        "reviewed_at": None,
    })
