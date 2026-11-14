from app.redaction.redactor import redact_interaction_fields


def test_email_is_redacted_with_stable_placeholder():
    fields = {
        "user_input": "Contact me at john@example.com about my order",
        "assistant_output": "I'll email john@example.com right away.",
    }
    outcome = redact_interaction_fields(fields)
    assert not outcome.quarantine
    assert "email" in outcome.pii_types
    assert "<EMAIL_1>" in outcome.redacted_fields["user_input"]
    # same raw value across fields maps to the same placeholder
    assert "<EMAIL_1>" in outcome.redacted_fields["assistant_output"]
    assert "john@example.com" not in outcome.redacted_fields["user_input"]
    assert "john@example.com" not in outcome.redacted_fields["assistant_output"]


def test_two_distinct_emails_get_distinct_placeholders():
    fields = {"user_input": "cc john@example.com and jane@example.com"}
    outcome = redact_interaction_fields(fields)
    assert "<EMAIL_1>" in outcome.redacted_fields["user_input"]
    assert "<EMAIL_2>" in outcome.redacted_fields["user_input"]


def test_phone_number_redacted():
    fields = {"user_input": "Call me at (216) 555-1234 tomorrow"}
    outcome = redact_interaction_fields(fields)
    assert "phone" in outcome.pii_types
    assert "555-1234" not in outcome.redacted_fields["user_input"]


def test_no_pii_leaves_text_untouched():
    fields = {"user_input": "What's the weather like today?"}
    outcome = redact_interaction_fields(fields)
    assert not outcome.pii_detected
    assert outcome.redacted_fields["user_input"] == "What's the weather like today?"


def test_api_key_triggers_quarantine_not_redaction():
    fields = {"user_input": "here is my key sk-abcdefghijklmnopqrstuvwxyz0123456789"}
    outcome = redact_interaction_fields(fields)
    assert outcome.quarantine
    assert outcome.quarantine_reason == "credential_or_private_key_detected"


def test_private_key_block_triggers_quarantine():
    fields = {
        "user_input": "-----BEGIN RSA PRIVATE KEY-----\nMIIBogIBAAJ...\n-----END RSA PRIVATE KEY-----"
    }
    outcome = redact_interaction_fields(fields)
    assert outcome.quarantine


def test_dense_pii_triggers_quarantine():
    fields = {
        "user_input": (
            "a@x.com b@x.com c@x.com d@x.com e@x.com "
            "216-555-0001 216-555-0002"
        )
    }
    outcome = redact_interaction_fields(fields)
    assert outcome.quarantine
    assert outcome.quarantine_reason == "dense_pii_concentration"


def test_none_fields_are_skipped_safely():
    fields = {"user_input": "hello", "system_prompt": None, "assistant_output": None}
    outcome = redact_interaction_fields(fields)
    assert "system_prompt" not in outcome.redacted_fields
    assert outcome.redacted_fields["user_input"] == "hello"
