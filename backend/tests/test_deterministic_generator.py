from datetime import datetime, timezone

from app.candidates.deterministic_generator import DeterministicCandidateGenerator
from app.candidates.interface import ClusterContext
from app.schemas.interaction import InteractionLog
from app.schemas.signal import SignalEvidence


def _log(iid, **overrides):
    base = dict(
        interaction_id=iid,
        timestamp=datetime.now(timezone.utc),
        application="support-bot",
        model_provider="anthropic",
        model_name="claude-sonnet",
        user_input="What is my balance?",
        assistant_output="$42.10",
        status="success",
        provenance="raw_log",
    )
    base.update(overrides)
    return InteractionLog(**base)


def _signal(iid, signal_type, source="deterministic"):
    return SignalEvidence(
        signal_id=f"sig_{iid}_{signal_type}",
        interaction_id=iid,
        signal_type=signal_type,
        source=source,
        score=None,
        evidence=["test"],
        created_at=datetime.now(timezone.utc),
    )


def test_generates_candidate_for_flagged_interaction():
    log = _log("a", user_feedback="negative")
    ctx = ClusterContext(
        cluster_id="c1",
        representative_logs=[log],
        signals_by_interaction={"a": [_signal("a", "thumbs_down")]},
        dominant_signal_types=["thumbs_down"],
    )
    candidates = DeterministicCandidateGenerator().generate(ctx)
    assert len(candidates) == 1
    c = candidates[0]
    assert c.review_status == "pending"
    assert c.reviewer_expected_output is None
    assert c.proposed_failure_category == "user_rated_negative"


def test_user_correction_becomes_generation_source():
    log = _log("a", user_correction="Actual balance is $124.50")
    ctx = ClusterContext(
        cluster_id="c1",
        representative_logs=[log],
        signals_by_interaction={"a": [_signal("a", "user_correction_present")]},
        dominant_signal_types=["user_correction_present"],
    )
    candidates = DeterministicCandidateGenerator().generate(ctx)
    assert candidates[0].generation_source == "user_correction"
    assert candidates[0].proposed_expected_output == "Actual balance is $124.50"


def test_no_signals_produces_no_candidate():
    log = _log("a")
    ctx = ClusterContext(
        cluster_id="c1", representative_logs=[log],
        signals_by_interaction={}, dominant_signal_types=[],
    )
    candidates = DeterministicCandidateGenerator().generate(ctx)
    assert candidates == []


def test_never_sets_review_status_other_than_pending():
    log = _log("a", status="error", error_type="RateLimitError")
    ctx = ClusterContext(
        cluster_id="c1", representative_logs=[log],
        signals_by_interaction={"a": [_signal("a", "http_or_api_error")]},
        dominant_signal_types=["http_or_api_error"],
    )
    candidates = DeterministicCandidateGenerator().generate(ctx)
    assert all(c.review_status == "pending" for c in candidates)
    assert all(c.reviewer_failure_category is None for c in candidates)
    assert all(c.reviewed_at is None for c in candidates)


def test_heuristic_only_signal_yields_heuristic_source_when_no_correction():
    log = _log("a", assistant_output="I can't help with that request.")
    ctx = ClusterContext(
        cluster_id="c1", representative_logs=[log],
        signals_by_interaction={"a": [_signal("a", "refusal_mismatch_suspected", source="heuristic")]},
        dominant_signal_types=["refusal_mismatch_suspected"],
    )
    candidates = DeterministicCandidateGenerator().generate(ctx)
    assert candidates[0].generation_source == "heuristic"
    assert candidates[0].proposed_failure_category == "suspected_incorrect_refusal"


def test_pure_latency_outlier_on_successful_content_is_groundable():
    """Content was fine (status=success), only latency was flagged — the
    observed output IS a legitimate expected output here, so this should
    NOT be left ungroundable."""
    log = _log("a", status="success", assistant_output="Your order ships tomorrow.")
    ctx = ClusterContext(
        cluster_id="c1", representative_logs=[log],
        signals_by_interaction={"a": [_signal("a", "latency_outlier", source="heuristic")]},
        dominant_signal_types=["latency_outlier"],
    )
    candidates = DeterministicCandidateGenerator().generate(ctx)
    assert candidates[0].proposed_expected_output == "Your order ships tomorrow."
    assert candidates[0].generation_source == "deterministic"


def test_retry_that_ultimately_succeeded_is_groundable():
    log = _log("a", status="success", retry_of="orig1", retry_count=1, assistant_output="Correct final answer.")
    ctx = ClusterContext(
        cluster_id="c1", representative_logs=[log],
        signals_by_interaction={"a": [_signal("a", "retry_occurred")]},
        dominant_signal_types=["retry_occurred"],
    )
    candidates = DeterministicCandidateGenerator().generate(ctx)
    assert candidates[0].proposed_expected_output == "Correct final answer."


def test_latency_outlier_combined_with_content_wrongness_signal_stays_ungroundable():
    """If a content-wrongness signal (e.g. thumbs_down) co-occurs with an
    operational signal, we must NOT assume the content was fine — that
    would be fabricating grounding that doesn't exist."""
    log = _log("a", status="success", user_feedback="negative", assistant_output="Some answer.")
    ctx = ClusterContext(
        cluster_id="c1", representative_logs=[log],
        signals_by_interaction={"a": [
            _signal("a", "latency_outlier", source="heuristic"),
            _signal("a", "thumbs_down"),
        ]},
        dominant_signal_types=["thumbs_down", "latency_outlier"],
    )
    candidates = DeterministicCandidateGenerator().generate(ctx)
    assert candidates[0].proposed_expected_output is None


def test_timeout_status_without_final_success_stays_ungroundable():
    """A record whose own status is 'timeout' (not success) never had a
    known-good final output — grounding must not be fabricated from a
    partial/failed response."""
    log = _log("a", status="timeout", assistant_output="Let me check on that for you.")
    ctx = ClusterContext(
        cluster_id="c1", representative_logs=[log],
        signals_by_interaction={"a": [_signal("a", "timeout")]},
        dominant_signal_types=["timeout"],
    )
    candidates = DeterministicCandidateGenerator().generate(ctx)
    assert candidates[0].proposed_expected_output is None
