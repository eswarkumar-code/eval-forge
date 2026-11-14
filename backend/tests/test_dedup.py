from app.ingestion.dedup import dedup_interactions
from app.ingestion.pipeline import ingest_batch


def _payload(iid, user_input="What is my order status?", **overrides):
    payload = dict(
        interaction_id=iid,
        timestamp="2026-08-01T10:00:00Z",
        application="support-bot",
        model_provider="anthropic",
        model_name="claude-sonnet",
        user_input=user_input,
        assistant_output="Your order ships tomorrow.",
        status="success",
    )
    payload.update(overrides)
    return payload


def test_exact_duplicate_detected():
    outcome = ingest_batch([_payload("a"), _payload("b")])  # identical content, different ids
    result = dedup_interactions(outcome.accepted)
    assert len(result.unique) == 1
    assert len(result.duplicates) == 1
    assert result.duplicate_of["b"] == "a"


def test_different_content_not_deduped():
    outcome = ingest_batch([_payload("a", user_input="question one"), _payload("b", user_input="question two")])
    result = dedup_interactions(outcome.accepted)
    assert len(result.unique) == 2
    assert len(result.duplicates) == 0


def test_three_way_duplicate_keeps_first_only():
    outcome = ingest_batch([_payload("a"), _payload("b"), _payload("c")])
    result = dedup_interactions(outcome.accepted)
    assert len(result.unique) == 1
    assert result.unique[0].interaction_id == "a"
    assert set(result.duplicate_of.keys()) == {"b", "c"}
    assert all(v == "a" for v in result.duplicate_of.values())


def test_empty_list_handled():
    result = dedup_interactions([])
    assert result.unique == []
    assert result.duplicates == []
