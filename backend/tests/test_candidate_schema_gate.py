from datetime import datetime, timezone

from app.schemas.candidate import EvalCandidate


def _bare_candidate(**overrides):
    base = dict(
        candidate_id="c1",
        source_interaction_ids=["i1"],
        input_text="some input",
        generation_source="deterministic",
        created_at=datetime.now(timezone.utc),
    )
    base.update(overrides)
    return EvalCandidate(**base)


def test_is_canonical_ready_false_for_blank_reviewer_expected_output_even_if_approved():
    """Schema-level defense in depth, independent of the review service:
    even if something else in the system ever sets review_status=approved
    with a blank reviewer_expected_output (bypassing approve_candidate),
    is_canonical_ready() must still return False."""
    c = _bare_candidate(
        review_status="approved",
        reviewer_expected_output="",
        reviewer_failure_category="some_category",
    )
    assert not c.is_canonical_ready()


def test_is_canonical_ready_false_for_whitespace_only_reviewer_category():
    c = _bare_candidate(
        review_status="approved",
        reviewer_expected_output="a valid answer",
        reviewer_failure_category="   ",
    )
    assert not c.is_canonical_ready()


def test_is_canonical_ready_true_for_genuinely_populated_reviewer_fields():
    c = _bare_candidate(
        review_status="approved",
        reviewer_expected_output="a valid answer",
        reviewer_failure_category="some_category",
    )
    assert c.is_canonical_ready()
