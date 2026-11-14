from datetime import datetime, timedelta, timezone

from app.schemas.interaction import InteractionLog
from app.signals.extractor import extract_signals


def _log(iid, **overrides):
    base = dict(
        interaction_id=iid,
        timestamp=datetime.now(timezone.utc),
        application="support-bot",
        model_provider="anthropic",
        model_name="claude-sonnet",
        user_input="What is my order status?",
        assistant_output="Your order ships tomorrow.",
        status="success",
        provenance="raw_log",
    )
    base.update(overrides)
    return InteractionLog(**base)


def _types(signals, interaction_id=None):
    return {s.signal_type for s in signals if interaction_id is None or s.interaction_id == interaction_id}


def test_thumbs_down_is_deterministic():
    log = _log("a", user_feedback="negative")
    result = extract_signals([log])
    matching = [s for s in result.signals if s.signal_type == "thumbs_down"]
    assert len(matching) == 1
    assert matching[0].source == "deterministic"


def test_error_status_produces_signal():
    log = _log("a", status="error", error_type="RateLimitError")
    result = extract_signals([log])
    assert "http_or_api_error" in _types(result.signals)


def test_no_signals_for_clean_interaction():
    log = _log("a")
    result = extract_signals([log])
    assert result.signals == []


def test_retry_occurred_flagged():
    logs = [_log("a"), _log("b", retry_of="a", retry_count=1)]
    result = extract_signals(logs)
    assert "retry_occurred" in _types(result.signals, "b")


def test_user_correction_and_response_edit_are_separate_signals():
    log = _log("a", user_correction="actually it's X", edited_response="corrected text")
    result = extract_signals([log])
    types = _types(result.signals)
    assert "user_correction_present" in types
    assert "response_edited" in types


def test_refusal_mismatch_heuristic_flags_refusal_language():
    log = _log("a", assistant_output="I can't help with that request.", status="success")
    result = extract_signals([log])
    matching = [s for s in result.signals if s.signal_type == "refusal_mismatch_suspected"]
    assert len(matching) == 1
    assert matching[0].source == "heuristic"
    assert matching[0].score is not None


def test_large_divergence_after_retry_detected():
    logs = [
        _log("a", assistant_output="Your order ships tomorrow via ground shipping."),
        _log("b", retry_of="a", assistant_output="I'm sorry, I don't have any information about orders at all."),
    ]
    result = extract_signals(logs)
    assert "large_divergence_after_retry" in _types(result.signals, "b")


def test_small_change_after_retry_not_flagged_as_divergence():
    logs = [
        _log("a", assistant_output="Your order ships tomorrow via ground shipping."),
        _log("b", retry_of="a", assistant_output="Your order ships tomorrow via ground shipping method."),
    ]
    result = extract_signals(logs)
    assert "large_divergence_after_retry" not in _types(result.signals, "b")


def test_latency_outlier_requires_min_sample_size():
    # only 3 records in the group -> below LATENCY_MIN_SAMPLE_SIZE, should not flag
    logs = [_log(f"a{i}", latency_ms=100.0) for i in range(3)]
    logs.append(_log("slow", latency_ms=5000.0))
    result = extract_signals(logs)
    assert "latency_outlier" not in _types(result.signals)


def test_latency_outlier_detected_with_sufficient_sample():
    logs = [_log(f"a{i}", latency_ms=100.0 + i) for i in range(10)]
    logs.append(_log("slow", latency_ms=8000.0))
    result = extract_signals(logs)
    matching = [s for s in result.signals if s.signal_type == "latency_outlier"]
    assert len(matching) == 1
    assert matching[0].interaction_id == "slow"
    assert matching[0].source == "heuristic"


def test_unsupported_claim_heuristic_flags_overconfident_unhedged_text():
    log = _log(
        "a",
        assistant_output="This treatment is definitely guaranteed to work 100% of the time for everyone.",
    )
    result = extract_signals([log])
    assert "unsupported_claim_suspected" in _types(result.signals)


def test_hedged_confident_language_not_flagged():
    log = _log(
        "a",
        assistant_output="I believe this treatment is generally effective, though results vary by individual.",
    )
    result = extract_signals([log])
    assert "unsupported_claim_suspected" not in _types(result.signals)


def test_malformed_structured_output_flagged():
    log = _log("a", status="success", structured_output_valid=False)
    result = extract_signals([log])
    assert "malformed_json_output" in _types(result.signals)
