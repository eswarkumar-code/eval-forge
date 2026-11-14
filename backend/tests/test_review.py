from datetime import datetime, timezone

import pytest

from app.review.service import (
    ReviewError, approve_candidate, edit_and_approve, reject_candidate, reset_to_pending,
)
from app.schemas.candidate import EvalCandidate


def _candidate(**overrides):
    base = dict(
        candidate_id="c1",
        source_interaction_ids=["i1"],
        input_text="What is my balance?",
        observed_output="$42.10",
        proposed_expected_output="$124.50",
        proposed_failure_category="user_corrected_response",
        generation_source="user_correction",
        created_at=datetime.now(timezone.utc),
    )
    base.update(overrides)
    return EvalCandidate(**base)


def test_pending_candidate_is_not_canonical_ready():
    c = _candidate()
    assert c.review_status == "pending"
    assert not c.is_canonical_ready()


def test_approve_accepting_proposed_fields_makes_canonical_ready():
    c = _candidate()
    approved = approve_candidate(c)
    assert approved.review_status == "approved"
    assert approved.reviewer_expected_output == "$124.50"
    assert approved.reviewer_failure_category == "user_corrected_response"
    assert approved.is_canonical_ready()
    assert approved.reviewed_at is not None


def test_approve_with_reviewer_override():
    c = _candidate()
    approved = approve_candidate(c, reviewer_expected_output="$999.99", reviewer_failure_category="billing_error")
    assert approved.reviewer_expected_output == "$999.99"
    assert approved.reviewer_failure_category == "billing_error"


def test_cannot_approve_candidate_with_no_expected_output_at_all():
    c = _candidate(proposed_expected_output=None)
    with pytest.raises(ReviewError):
        approve_candidate(c)


def test_cannot_approve_candidate_with_no_category_at_all():
    c = _candidate(proposed_failure_category=None)
    with pytest.raises(ReviewError):
        approve_candidate(c)


def test_cannot_approve_candidate_with_empty_string_expected_output():
    """Found via Gate 1: an is-None-only check would let this through
    since '' is not None. The gate must reject blank strings too."""
    c = _candidate(proposed_expected_output="")
    with pytest.raises(ReviewError):
        approve_candidate(c)


def test_cannot_approve_candidate_with_whitespace_only_expected_output():
    c = _candidate(proposed_expected_output="   \n  ")
    with pytest.raises(ReviewError):
        approve_candidate(c)


def test_cannot_approve_candidate_with_empty_string_category():
    c = _candidate(proposed_failure_category="")
    with pytest.raises(ReviewError):
        approve_candidate(c)


def test_reject_does_not_set_reviewer_fields():
    c = _candidate()
    rejected = reject_candidate(c, reviewer_notes="not a real failure")
    assert rejected.review_status == "rejected"
    assert rejected.reviewer_expected_output is None
    assert not rejected.is_canonical_ready()


def test_edit_and_approve_requires_both_fields():
    c = _candidate()
    with pytest.raises(ReviewError):
        edit_and_approve(c, reviewer_expected_output="", reviewer_failure_category="x")


def test_edit_and_approve_success():
    c = _candidate()
    approved = edit_and_approve(c, reviewer_expected_output="corrected answer", reviewer_failure_category="new_category")
    assert approved.is_canonical_ready()
    assert approved.reviewer_expected_output == "corrected answer"


def test_reset_to_pending_clears_reviewer_fields():
    c = _candidate()
    approved = approve_candidate(c)
    reset = reset_to_pending(approved)
    assert reset.review_status == "pending"
    assert reset.reviewer_expected_output is None
    assert reset.reviewed_at is None


def test_original_candidate_object_never_mutated():
    """model_copy must not mutate the original — pending candidates stay
    pending even after we approve a copy of them."""
    c = _candidate()
    approve_candidate(c)
    assert c.review_status == "pending"
    assert c.reviewer_expected_output is None
