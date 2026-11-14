from app.ingestion.pipeline import ingest_batch


def _valid_payload(**overrides):
    payload = dict(
        interaction_id="i1",
        timestamp="2026-08-01T10:00:00Z",
        application="support-bot",
        model_provider="anthropic",
        model_name="claude-sonnet",
        user_input="What is my order status?",
        assistant_output="Your order ships tomorrow.",
        status="success",
        user_feedback="positive",
    )
    payload.update(overrides)
    return payload


def test_valid_record_is_accepted():
    outcome = ingest_batch([_valid_payload()])
    assert len(outcome.accepted) == 1
    assert len(outcome.quarantined) == 0
    assert outcome.accepted[0].interaction_id == "i1"


def test_missing_required_field_is_quarantined_not_dropped_silently():
    payload = _valid_payload()
    del payload["user_input"]
    outcome = ingest_batch([payload])
    assert len(outcome.accepted) == 0
    assert len(outcome.quarantined) == 1
    assert "user_input" in outcome.quarantined[0].error_detail


def test_completely_malformed_payload_is_quarantined():
    outcome = ingest_batch([{"this_is": "not a real log record", "nested": {"junk": [1, 2, 3]}}])
    assert len(outcome.accepted) == 0
    assert len(outcome.quarantined) == 1


def test_non_dict_payload_does_not_crash_batch():
    outcome = ingest_batch(["just a string", 42, None, _valid_payload(interaction_id="ok1")])
    assert len(outcome.accepted) == 1
    assert outcome.accepted[0].interaction_id == "ok1"
    assert len(outcome.quarantined) == 3


def test_invalid_status_value_quarantined():
    outcome = ingest_batch([_valid_payload(status="totally_bogus_status")])
    assert len(outcome.quarantined) == 1
    assert "status" in outcome.quarantined[0].error_detail


def test_unparseable_timestamp_quarantined():
    outcome = ingest_batch([_valid_payload(timestamp="not-a-real-date")])
    assert len(outcome.quarantined) == 1


def test_pii_is_redacted_during_ingestion_not_after():
    outcome = ingest_batch(
        [_valid_payload(interaction_id="i2", user_input="my email is jane@example.com")]
    )
    assert len(outcome.accepted) == 1
    log = outcome.accepted[0]
    assert "jane@example.com" not in log.user_input
    assert log.pii_detected is True
    assert "email" in log.pii_types


def test_credential_in_payload_is_quarantined_never_stored():
    outcome = ingest_batch(
        [_valid_payload(interaction_id="i3", user_input="here's my key sk-abcdefghijklmnopqrstuvwxyz0123456789")]
    )
    assert len(outcome.accepted) == 0
    assert len(outcome.quarantined) == 1
    assert "sensitive_data" in outcome.quarantined[0].reason


def test_batch_of_mixed_good_and_bad_records_partial_success():
    payloads = [
        _valid_payload(interaction_id="good1"),
        {"garbage": True},
        _valid_payload(interaction_id="good2", status="bogus"),
        _valid_payload(interaction_id="good3"),
    ]
    outcome = ingest_batch(payloads)
    accepted_ids = {log.interaction_id for log in outcome.accepted}
    assert accepted_ids == {"good1", "good3"}
    assert len(outcome.quarantined) == 2


def test_content_hash_populated_for_dedup():
    outcome = ingest_batch([_valid_payload()])
    assert outcome.accepted[0].content_hash is not None
    assert len(outcome.accepted[0].content_hash) == 64  # sha256 hex


def test_retry_and_correction_fields_pass_through_distinctly():
    outcome = ingest_batch(
        [
            _valid_payload(interaction_id="r1", retry_of="i1", retry_count=1),
            _valid_payload(interaction_id="c1", user_correction="Actually the answer is Y."),
        ]
    )
    by_id = {log.interaction_id: log for log in outcome.accepted}
    assert by_id["r1"].retry_of == "i1"
    assert by_id["r1"].user_correction is None
    assert by_id["c1"].retry_of is None
    assert by_id["c1"].user_correction is not None
