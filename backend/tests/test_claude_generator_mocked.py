import json
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from app.candidates.claude_generator import ClaudeCandidateGenerator
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
        user_input="If today is March 3rd and my subscription renews in 30 days, what date is that?",
        assistant_output="Your subscription will renew on March 25th.",
        status="success",
        provenance="raw_log",
    )
    base.update(overrides)
    return InteractionLog(**base)


def _signal(iid, signal_type="thumbs_down"):
    return SignalEvidence(
        signal_id=f"sig_{iid}", interaction_id=iid, signal_type=signal_type,
        source="deterministic", score=None, evidence=["test"],
        created_at=datetime.now(timezone.utc),
    )


class FakeAnthropicClient:
    """Test double implementing the same minimal surface as
    _RealAnthropicClientAdapter — no network, no SDK, no API key."""

    def __init__(self, response_text: str, raise_error: Exception | None = None):
        self._response_text = response_text
        self._raise_error = raise_error
        self.calls = []

    def messages_create(self, **kwargs):
        self.calls.append(kwargs)
        if self._raise_error:
            raise self._raise_error
        return SimpleNamespace(content=[SimpleNamespace(text=self._response_text)])


def _ctx(log, signal_type="thumbs_down"):
    return ClusterContext(
        cluster_id="c1",
        representative_logs=[log],
        signals_by_interaction={log.interaction_id: [_signal(log.interaction_id, signal_type)]},
        dominant_signal_types=[signal_type],
    )


def test_valid_model_response_produces_model_generated_candidate():
    response_json = json.dumps({
        "failure_category": "incorrect_date_reasoning",
        "expected_output": "Your subscription will renew on April 2nd.",
        "difficulty": "medium",
        "severity": "high",
        "reasoning": "30 days after March 3rd is April 2nd, not March 25th.",
    })
    client = FakeAnthropicClient(response_json)
    log = _log("a")
    candidates = ClaudeCandidateGenerator(client).generate(_ctx(log))

    assert len(candidates) == 1
    c = candidates[0]
    assert c.generation_source == "model_generated"
    assert c.proposed_failure_category == "incorrect_date_reasoning"
    assert c.proposed_expected_output == "Your subscription will renew on April 2nd."
    assert c.review_status == "pending"
    assert c.reviewer_expected_output is None
    assert c.reviewer_failure_category is None
    assert len(client.calls) == 1


def test_never_sets_review_status_or_reviewer_fields_regardless_of_model_output():
    response_json = json.dumps({
        "failure_category": "wrong_classification",
        "expected_output": "billing_dispute",
        "difficulty": "easy",
        "severity": "medium",
        "reasoning": "should be billing",
    })
    client = FakeAnthropicClient(response_json)
    candidates = ClaudeCandidateGenerator(client).generate(_ctx(_log("a")))
    assert candidates[0].review_status == "pending"
    assert candidates[0].reviewed_at is None


def test_malformed_json_response_skips_without_crashing():
    client = FakeAnthropicClient("this is not json at all {{{")
    candidates = ClaudeCandidateGenerator(client).generate(_ctx(_log("a")))
    assert candidates == []


def test_markdown_fenced_json_response_is_parsed_correctly():
    """Real-world models (including live Claude Sonnet 4.5, confirmed via
    the live acceptance suite) commonly wrap JSON in ```json fences even
    when told not to. This must be handled as a normal response, not
    treated as malformed output."""
    fenced = "```json\n" + json.dumps({
        "failure_category": "incorrect_date_reasoning",
        "expected_output": "Your subscription will renew on April 2nd.",
        "difficulty": "medium",
        "severity": "high",
        "reasoning": "30 days after March 3rd is April 2nd.",
    }) + "\n```"
    client = FakeAnthropicClient(fenced)
    candidates = ClaudeCandidateGenerator(client).generate(_ctx(_log("a")))
    assert len(candidates) == 1
    assert candidates[0].proposed_failure_category == "incorrect_date_reasoning"
    assert candidates[0].proposed_expected_output == "Your subscription will renew on April 2nd."


def test_fenced_json_without_language_tag_is_parsed():
    fenced = "```\n" + json.dumps({
        "failure_category": "wrong_classification",
        "expected_output": "billing_dispute",
        "difficulty": "easy",
        "severity": "medium",
        "reasoning": "n/a",
    }) + "\n```"
    client = FakeAnthropicClient(fenced)
    candidates = ClaudeCandidateGenerator(client).generate(_ctx(_log("a")))
    assert len(candidates) == 1
    assert candidates[0].proposed_failure_category == "wrong_classification"


def test_genuinely_malformed_fenced_content_still_skips_honestly():
    """A fence around genuinely broken JSON must still fail to parse and
    be skipped -- fence-stripping must not mask real malformed output."""
    fenced = "```json\nthis is not valid json {{{\n```"
    client = FakeAnthropicClient(fenced)
    candidates = ClaudeCandidateGenerator(client).generate(_ctx(_log("a")))
    assert candidates == []


def test_missing_failure_category_is_skipped():
    response_json = json.dumps({"expected_output": "something", "difficulty": "easy", "severity": "low"})
    client = FakeAnthropicClient(response_json)
    candidates = ClaudeCandidateGenerator(client).generate(_ctx(_log("a")))
    assert candidates == []


def test_empty_string_expected_output_normalized_to_none_not_left_as_blank_string():
    """Found via Gate 1 live acceptance testing: live Claude Sonnet 4.5
    returned expected_output='' (empty string) rather than JSON null for
    an ungroundable case. This must be normalized to None, not passed
    through as a blank string that could silently satisfy an `is not None`
    check downstream."""
    response_json = json.dumps({
        "failure_category": "overconfident_unsupported_claim",
        "expected_output": "",
        "difficulty": "hard",
        "severity": "high",
        "reasoning": "no way to know the correct answer from context alone",
    })
    client = FakeAnthropicClient(response_json)
    candidates = ClaudeCandidateGenerator(client).generate(_ctx(_log("a")))
    assert len(candidates) == 1
    assert candidates[0].proposed_expected_output is None


def test_whitespace_only_expected_output_normalized_to_none():
    response_json = json.dumps({
        "failure_category": "some_category",
        "expected_output": "   \n  ",
        "difficulty": "medium",
        "severity": "medium",
        "reasoning": "n/a",
    })
    client = FakeAnthropicClient(response_json)
    candidates = ClaudeCandidateGenerator(client).generate(_ctx(_log("a")))
    assert candidates[0].proposed_expected_output is None


def test_non_string_expected_output_normalized_to_none():
    """Defends against a model returning e.g. a number or list where a
    string was expected."""
    response_json = json.dumps({
        "failure_category": "some_category",
        "expected_output": 42,
        "difficulty": "medium",
        "severity": "medium",
        "reasoning": "n/a",
    })
    client = FakeAnthropicClient(response_json)
    candidates = ClaudeCandidateGenerator(client).generate(_ctx(_log("a")))
    assert candidates[0].proposed_expected_output is None


def test_null_expected_output_is_honest_not_fabricated():
    response_json = json.dumps({
        "failure_category": "ambiguous_request",
        "expected_output": None,
        "difficulty": "hard",
        "severity": "low",
        "reasoning": "not enough context to know the right answer",
    })
    client = FakeAnthropicClient(response_json)
    candidates = ClaudeCandidateGenerator(client).generate(_ctx(_log("a")))
    assert candidates[0].proposed_expected_output is None
    assert candidates[0].proposed_failure_category == "ambiguous_request"


def test_api_error_skips_that_interaction_without_crashing_batch():
    client = FakeAnthropicClient("", raise_error=RuntimeError("simulated network failure"))
    candidates = ClaudeCandidateGenerator(client).generate(_ctx(_log("a")))
    assert candidates == []


def test_invalid_difficulty_severity_default_to_medium():
    response_json = json.dumps({
        "failure_category": "some_category",
        "expected_output": "x",
        "difficulty": "super_hard",  # invalid
        "severity": "extreme",  # invalid
        "reasoning": "n/a",
    })
    client = FakeAnthropicClient(response_json)
    candidates = ClaudeCandidateGenerator(client).generate(_ctx(_log("a")))
    assert candidates[0].difficulty == "medium"
    assert candidates[0].severity == "medium"


def test_respects_max_representatives_cap():
    logs = [_log(f"i{i}") for i in range(10)]
    ctx = ClusterContext(
        cluster_id="c1",
        representative_logs=logs,
        signals_by_interaction={log.interaction_id: [_signal(log.interaction_id)] for log in logs},
        dominant_signal_types=["thumbs_down"],
    )
    response_json = json.dumps({
        "failure_category": "x", "expected_output": "y", "difficulty": "easy", "severity": "low", "reasoning": "n/a",
    })
    client = FakeAnthropicClient(response_json)
    candidates = ClaudeCandidateGenerator(client).generate(ctx)
    assert len(client.calls) <= 3  # MAX_REPRESENTATIVES_PER_CLUSTER
    assert len(candidates) == len(client.calls)
