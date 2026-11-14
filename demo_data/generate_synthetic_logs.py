"""
Generates a realistic synthetic production-log dataset with known planted
failure themes, per design_lock.md section 4.

Output: demo_data/synthetic_logs.jsonl (raw ingest format — as if pulled
straight from production, i.e. INCLUDES some raw PII on purpose, so the
ingestion/redaction layer has real material to prove itself against).

Each record's metadata carries a "_planted_theme" field. This is
ground-truth-for-evaluation-purposes ONLY. It must never be treated as a
signal, a feature for clustering, or copied into the canonical eval
dataset — it exists solely so cluster_quality.py can compute purity
against something real. The clustering code itself must never read
_planted_theme.

Planted themes (from design_lock.md section 4):
  - incorrect_date_reasoning
  - malformed_structured_json
  - refusal_when_answer_should_be_allowed
  - unsupported_factual_claims
  - missing_required_fields
  - wrong_classification
  - excessive_latency_or_retry
  - incomplete_multi_part_answer

Plus normal noise: successful interactions, ambiguous requests, exact
duplicates, a few PII-bearing records, and a handful of adversarial/weird
records for ingestion robustness testing.
"""
from __future__ import annotations

import json
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

random.seed(42)

OUTPUT_PATH = Path(__file__).parent / "synthetic_logs.jsonl"

APPLICATIONS = ["support-bot", "docs-assistant", "billing-helper"]
MODELS = [("anthropic", "claude-sonnet"), ("anthropic", "claude-haiku"), ("openai", "gpt-4o-mini")]

START_TIME = datetime(2026, 7, 1, tzinfo=timezone.utc)


def _ts(offset_minutes: int) -> str:
    return (START_TIME + timedelta(minutes=offset_minutes)).isoformat().replace("+00:00", "Z")


def _base_record(offset, **overrides):
    app, feature = random.choice([
        ("support-bot", "order_status"),
        ("support-bot", "returns"),
        ("docs-assistant", "api_reference"),
        ("billing-helper", "invoice_lookup"),
    ])
    provider, model = random.choice(MODELS)
    rec = dict(
        interaction_id=str(uuid4()),
        conversation_id=None,
        parent_interaction_id=None,
        timestamp=_ts(offset),
        application=app,
        feature=feature,
        model_provider=provider,
        model_name=model,
        system_prompt="You are a helpful assistant for our product.",
        user_input="",
        assistant_output="",
        input_tokens=random.randint(20, 200),
        output_tokens=random.randint(20, 300),
        latency_ms=round(random.gauss(450, 80), 1),
        status="success",
        user_feedback="none",
        retry_of=None,
        retry_count=0,
        user_correction=None,
        edited_response=None,
        error_type=None,
        error_message=None,
        structured_output_valid=None,
        metadata={},
    )
    rec.update(overrides)
    return rec


def _plant(rec: dict, theme: str) -> dict:
    rec["metadata"]["_planted_theme"] = theme
    return rec


records: list[dict] = []
offset = 0


def add(rec):
    global offset
    offset += random.randint(1, 5)
    records.append(rec)


# ---- 1. Successful, unremarkable interactions (the majority class) ----
SUCCESS_PAIRS = [
    ("What's the status of my order #4471?", "Your order #4471 shipped yesterday and should arrive within 3 business days."),
    ("How do I reset my password?", "Go to Settings > Security > Reset Password and follow the emailed link."),
    ("What's included in the API reference for the /users endpoint?", "The /users endpoint returns id, name, email, and created_at fields."),
    ("Can I get a copy of my last invoice?", "Sure — your most recent invoice (INV-2291) is attached to your account under Billing."),
    ("Do you support bulk exports?", "Yes, bulk exports are available in CSV and JSON from the Data page."),
]
for _ in range(40):
    q, a = random.choice(SUCCESS_PAIRS)
    add(_base_record(offset, user_input=q, assistant_output=a, status="success", user_feedback=random.choice(["positive", "none", "none"])))

# ---- 2. Planted theme: incorrect_date_reasoning ----
DATE_CASES = [
    ("If today is March 3rd and my subscription renews in 30 days, what date does it renew?",
     "Your subscription will renew on March 25th."),  # wrong: should be April 2nd
    ("My trial started on Jan 15 and lasts 14 days, when does it end?",
     "Your trial ends on Jan 21."),  # wrong: should be Jan 29
]
for i in range(6):
    q, a = DATE_CASES[i % len(DATE_CASES)]
    rec = _base_record(offset, user_input=q, assistant_output=a, status="success", user_feedback="negative")
    add(_plant(rec, "incorrect_date_reasoning"))

# ---- 3. Planted theme: malformed_structured_json ----
for i in range(6):
    rec = _base_record(
        offset,
        user_input="Return the order summary as JSON with fields order_id, total, status.",
        assistant_output='{"order_id": "4471", "total": 59.99 "status": "shipped"}',  # missing comma
        status="malformed_output",
        structured_output_valid=False,
    )
    add(_plant(rec, "malformed_structured_json"))

# ---- 4. Planted theme: refusal_when_answer_should_be_allowed ----
BENIGN_REFUSED = [
    "What are your standard business hours?",
    "Can you tell me the price of the Pro plan?",
    "What's the difference between the Basic and Pro tiers?",
]
for i in range(6):
    q = BENIGN_REFUSED[i % len(BENIGN_REFUSED)]
    rec = _base_record(
        offset, user_input=q,
        assistant_output="I can't help with that request.",
        status="success", user_feedback="negative",
    )
    add(_plant(rec, "refusal_when_answer_should_be_allowed"))

# ---- 5. Planted theme: unsupported_factual_claims ----
for i in range(6):
    rec = _base_record(
        offset,
        user_input="Will this integration work with our legacy system?",
        assistant_output="Yes, this definitely works with 100% of legacy systems and is guaranteed to have zero issues.",
        status="success", user_feedback="negative",
    )
    add(_plant(rec, "unsupported_factual_claims"))

# ---- 6. Planted theme: missing_required_fields ----
for i in range(5):
    rec = _base_record(
        offset,
        user_input="Summarize my account details as structured data.",
        assistant_output='{"name": "Jordan Blake"}',  # missing plan/status/etc fields the app expects
        status="success", structured_output_valid=False,
    )
    add(_plant(rec, "missing_required_fields"))

# ---- 7. Planted theme: wrong_classification ----
for i in range(6):
    rec = _base_record(
        offset,
        user_input="I was charged twice this month, is this a billing question or a bug report?",
        assistant_output="This has been classified as: General Feedback.",  # wrong: should be billing_dispute
        status="success", user_feedback="negative",
    )
    add(_plant(rec, "wrong_classification"))

# ---- 8. Planted theme: excessive_latency_or_retry ----
for i in range(6):
    rec = _base_record(
        offset,
        user_input="What's the status of my refund request?",
        assistant_output="Your refund is being processed.",
        status="success", latency_ms=round(random.uniform(6000, 9500), 1),
    )
    add(_plant(rec, "excessive_latency_or_retry"))
for i in range(4):
    orig_id = str(uuid4())
    orig = _base_record(
        offset, interaction_id=orig_id,
        user_input="What's the status of my refund request?",
        assistant_output="Let me check on that for you.",
        status="timeout",
    )
    add(_plant(orig, "excessive_latency_or_retry"))
    retry = _base_record(
        offset, retry_of=orig_id, retry_count=1,
        user_input="What's the status of my refund request?",
        assistant_output="Your refund is being processed.",
        status="success",
    )
    add(_plant(retry, "excessive_latency_or_retry"))

# ---- 9. Planted theme: incomplete_multi_part_answer ----
for i in range(6):
    rec = _base_record(
        offset,
        user_input="Can you tell me (1) my current plan, (2) my renewal date, and (3) how to upgrade?",
        assistant_output="Your current plan is Pro.",  # only answers part 1
        status="success", user_feedback="negative",
    )
    add(_plant(rec, "incomplete_multi_part_answer"))

# ---- 10. Ambiguous requests (noise, not a planted failure) ----
AMBIGUOUS = [
    "It's not working.",
    "Can you fix this?",
    "Same issue as before.",
]
for i in range(8):
    q = random.choice(AMBIGUOUS)
    add(_base_record(offset, user_input=q, assistant_output="Could you share more detail about what's happening?", status="success"))

# ---- 11. User corrections (deterministic signal source material) ----
for i in range(5):
    add(_base_record(
        offset,
        user_input="What's my current balance?",
        assistant_output="Your balance is $42.10.",
        status="success",
        user_correction="That's wrong, my balance is actually $124.50.",
        user_feedback="negative",
    ))

# ---- 12. PII-bearing records (raw on purpose — tests redaction) ----
PII_CASES = [
    ("My email is jordan.blake@example.com, can you update it on file?", "Sure, I've noted jordan.blake@example.com on your account."),
    ("Please call me back at (216) 555-0148 about my order.", "I'll have someone call (216) 555-0148 shortly."),
]
for i in range(6):
    q, a = random.choice(PII_CASES)
    add(_base_record(offset, user_input=q, assistant_output=a, status="success"))

# ---- 13. Exact duplicates (dedup test material) ----
dup_q, dup_a = "How do I cancel my subscription?", "Go to Settings > Billing > Cancel Subscription."
for i in range(4):
    add(_base_record(offset, application="billing-helper", feature="invoice_lookup",
                      user_input=dup_q, assistant_output=dup_a, status="success"))

# ---- 14. Adversarial / weird / malformed records for ingestion robustness ----
adversarial = [
    {"garbage_field": "not a real record", "nested": {"a": [1, 2, {"b": None}]}},
    {"interaction_id": "weird1", "application": "support-bot"},  # missing user_input etc
    {"interaction_id": "weird2", "timestamp": "not-a-date", "application": "support-bot",
     "model_provider": "anthropic", "model_name": "claude-sonnet", "user_input": "hi", "status": "success"},
    {"interaction_id": "weird3", "application": "support-bot", "model_provider": "anthropic",
     "model_name": "claude-sonnet", "user_input": "hi", "status": "not_a_real_status"},
    "just a raw string, not even an object",
    42,
    None,
    {"interaction_id": "weird4", "application": "support-bot", "model_provider": "anthropic",
     "model_name": "claude-sonnet", "user_input": "my key is sk-abcdefghijklmnopqrstuvwxyz0123456789", "status": "success"},
    {"interaction_id": "weird5", "application": "support-bot", "model_provider": "anthropic",
     "model_name": "claude-sonnet",
     "user_input": "contact a@x.com b@x.com c@x.com d@x.com e@x.com 216-555-0001 216-555-0002",
     "status": "success"},
]
records.extend(adversarial)

random.shuffle(records)

with OUTPUT_PATH.open("w") as f:
    for r in records:
        f.write(json.dumps(r, default=str) + "\n")

print(f"Wrote {len(records)} records to {OUTPUT_PATH}")
