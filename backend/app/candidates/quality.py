"""
Candidate-quality checks, per design_lock.md section 5:
  reproducible, specific, groundable, non-duplicative, representative,
  reviewable, properly sourced.

These are NOT a gate that silently deletes candidates — they annotate
each candidate with a quality_flags list so a human reviewer sees them
before approving/rejecting. Only a human decides whether a flagged
candidate is still worth approving.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from difflib import SequenceMatcher

from app.schemas.candidate import EvalCandidate

DUPLICATE_SIMILARITY_THRESHOLD = 0.9
MIN_INPUT_LENGTH_FOR_SPECIFICITY = 8  # characters; catches near-empty inputs


@dataclass
class CandidateQualityFlags:
    candidate_id: str
    flags: list[str] = field(default_factory=list)

    @property
    def is_clean(self) -> bool:
        return len(self.flags) == 0


def _is_duplicate_of_any(candidate: EvalCandidate, existing: list[EvalCandidate]) -> bool:
    for other in existing:
        if other.candidate_id == candidate.candidate_id:
            continue
        ratio = SequenceMatcher(None, candidate.input_text, other.input_text).ratio()
        if ratio >= DUPLICATE_SIMILARITY_THRESHOLD:
            return True
    return False


def evaluate_candidate_quality(
    candidate: EvalCandidate,
    existing_canonical: list[EvalCandidate],
) -> CandidateQualityFlags:
    flags: list[str] = []

    # reproducible: must have at least one source interaction and the
    # original input text present
    if not candidate.source_interaction_ids or not candidate.input_text:
        flags.append("not_reproducible_missing_source_or_input")

    # specific: not a near-empty / trivially vague input
    if candidate.input_text and len(candidate.input_text.strip()) < MIN_INPUT_LENGTH_FOR_SPECIFICITY:
        flags.append("input_too_short_to_be_specific")

    # groundable: must have SOME basis for an expected output, OR
    # explicitly have none proposed (honest null is fine; a category with
    # no observed_output AND no proposed_expected_output is weak)
    if not candidate.observed_output and not candidate.proposed_expected_output:
        flags.append("not_groundable_no_observed_or_expected_output")

    # non-duplicative
    if _is_duplicate_of_any(candidate, existing_canonical):
        flags.append("likely_duplicate_of_existing_canonical_case")

    # reviewable: must have a failure category proposed at all, and
    # observed_output present so a reviewer isn't reconstructing history
    if not candidate.proposed_failure_category:
        flags.append("not_reviewable_missing_proposed_category")
    if not candidate.observed_output:
        flags.append("not_reviewable_missing_observed_output")

    # properly sourced: generation_source must be one of the declared
    # honest values (schema already enforces this, but double-check here
    # since this module is meant to be the "is this candidate any good"
    # single source of truth)
    if candidate.generation_source not in ("user_correction", "deterministic", "heuristic", "model_generated"):
        flags.append("improperly_sourced")  # pragma: no cover - schema prevents this

    return CandidateQualityFlags(candidate_id=candidate.candidate_id, flags=flags)
