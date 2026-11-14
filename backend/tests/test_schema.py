from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from app.schemas.interaction import InteractionLog


def _base(**overrides):
    fields = dict(
        interaction_id="int_1",
        timestamp=datetime.now(timezone.utc),
        application="support-bot",
        model_provider="anthropic",
        model_name="claude-sonnet",
        user_input="What is my order status?",
        status="success",
        provenance="raw_log",
    )
    fields.update(overrides)
    return fields


def test_minimal_valid_record():
    log = InteractionLog(**_base())
    assert log.retry_count == 0
    assert log.user_feedback == "none"
    assert log.pii_detected is False


def test_retry_of_cannot_reference_self():
    with pytest.raises(ValidationError):
        InteractionLog(**_base(interaction_id="int_1", retry_of="int_1"))


def test_retry_of_different_interaction_is_fine():
    log = InteractionLog(**_base(interaction_id="int_2", retry_of="int_1"))
    assert log.retry_of == "int_1"


def test_negative_retry_count_rejected():
    with pytest.raises(ValidationError):
        InteractionLog(**_base(retry_count=-1))


def test_invalid_status_rejected():
    with pytest.raises(ValidationError):
        InteractionLog(**_base(status="not_a_real_status"))


def test_conversation_fields_preserve_sequence_not_collapsed():
    turn1 = InteractionLog(**_base(interaction_id="c1_t1", conversation_id="c1"))
    turn2 = InteractionLog(
        **_base(
            interaction_id="c1_t2",
            conversation_id="c1",
            parent_interaction_id="c1_t1",
            user_input="follow up question",
        )
    )
    assert turn2.conversation_id == turn1.conversation_id
    assert turn2.parent_interaction_id == turn1.interaction_id
    # each turn remains an independent record, not merged into one string
    assert turn1.user_input != turn2.user_input


def test_retry_and_correction_are_distinct_concepts():
    retry = InteractionLog(**_base(interaction_id="i2", retry_of="i1", retry_count=1))
    correction = InteractionLog(
        **_base(interaction_id="i3", user_correction="The correct answer is X.")
    )
    assert retry.user_correction is None
    assert correction.retry_of is None
