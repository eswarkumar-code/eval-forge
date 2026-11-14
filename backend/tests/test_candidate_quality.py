from datetime import datetime, timezone

from app.candidates.quality import evaluate_candidate_quality
from app.schemas.candidate import EvalCandidate


def _candidate(cid, **overrides):
    base = dict(
        candidate_id=cid,
        source_interaction_ids=["i1"],
        input_text="What is my subscription renewal date?",
        observed_output="It renews on the 25th.",
        proposed_expected_output="It renews on the 2nd.",
        proposed_failure_category="incorrect_date_reasoning",
        generation_source="deterministic",
        created_at=datetime.now(timezone.utc),
    )
    base.update(overrides)
    return EvalCandidate(**base)


def test_clean_candidate_has_no_flags():
    c = _candidate("c1")
    result = evaluate_candidate_quality(c, existing_canonical=[])
    assert result.is_clean


def test_missing_source_interaction_flagged():
    c = _candidate("c1", source_interaction_ids=[])
    result = evaluate_candidate_quality(c, existing_canonical=[])
    assert "not_reproducible_missing_source_or_input" in result.flags


def test_too_short_input_flagged():
    c = _candidate("c1", input_text="hi")
    result = evaluate_candidate_quality(c, existing_canonical=[])
    assert "input_too_short_to_be_specific" in result.flags


def test_no_grounding_basis_flagged():
    c = _candidate("c1", observed_output=None, proposed_expected_output=None)
    result = evaluate_candidate_quality(c, existing_canonical=[])
    assert "not_groundable_no_observed_or_expected_output" in result.flags


def test_duplicate_of_existing_canonical_flagged():
    existing = _candidate("existing1", input_text="What is my subscription renewal date?")
    new = _candidate("c2", input_text="What is my subscription renewal date?")
    result = evaluate_candidate_quality(new, existing_canonical=[existing])
    assert "likely_duplicate_of_existing_canonical_case" in result.flags


def test_dissimilar_input_not_flagged_as_duplicate():
    existing = _candidate("existing1", input_text="What is my subscription renewal date?")
    new = _candidate("c2", input_text="How do I cancel my account entirely?")
    result = evaluate_candidate_quality(new, existing_canonical=[existing])
    assert "likely_duplicate_of_existing_canonical_case" not in result.flags


def test_missing_proposed_category_flagged():
    c = _candidate("c1", proposed_failure_category=None)
    result = evaluate_candidate_quality(c, existing_canonical=[])
    assert "not_reviewable_missing_proposed_category" in result.flags


def test_missing_observed_output_flagged():
    c = _candidate("c1", observed_output=None)
    result = evaluate_candidate_quality(c, existing_canonical=[])
    assert "not_reviewable_missing_observed_output" in result.flags
