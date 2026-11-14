"""
Claude-backed candidate generator.

Calls the Anthropic API to propose a failure category and expected/ideal
output for a representative interaction in a cluster. This is genuinely
model-assisted — NOT a template — but its output is never treated as
ground truth: generation_source is always "model_generated", and the
base class contract (enforced by _assert_generator_contract) guarantees
review_status stays "pending" and reviewer_* fields stay unset no matter
what the model returns.

Testability:
  - The Anthropic client is injected (constructor param), not constructed
    internally, specifically so the main automated test suite can pass a
    mock/fake client and never make a real network call or require an
    API key.
  - A separate, opt-in live acceptance suite (tests_live/) constructs a
    real anthropic.Anthropic() client and exercises this class against
    the actual API using synthetic/redacted records only.

Failure handling: if the API call fails or returns unparseable output,
this generator degrades to producing NO candidate for that interaction
(logged) rather than fabricating a plausible-looking but fictitious
category/expected-output pair.
"""
from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from typing import Any, Protocol
from uuid import uuid4

from app.candidates.interface import CandidateGenerator, ClusterContext
from app.schemas.candidate import Difficulty, EvalCandidate, Severity

logger = logging.getLogger(__name__)

MODEL_NAME = "claude-sonnet-4-5"
MAX_REPRESENTATIVES_PER_CLUSTER = 3  # keep API usage proportional; cluster is already a group of similar failures

_SYSTEM_PROMPT = """You are assisting an AI evaluation engineer in triaging production LLM failures.
You will be shown a REDACTED production interaction (user input, the model's actual output, and \
which automated failure signals were attached to it). Sensitive data has already been replaced \
with placeholders like <EMAIL_1> — treat those as opaque tokens, not real values.

Your job is to propose:
  1. A short failure_category (snake_case, e.g. "incorrect_date_reasoning", "suspected_hallucination", \
"incomplete_multi_part_answer") that best names the underlying problem.
  2. A proposed expected_output: what a good response would have looked like. If you cannot responsibly \
determine what the correct answer should be from the given context alone, set expected_output to null \
rather than guessing.
  3. A difficulty rating: "easy", "medium", or "hard" (how hard is this case to get right).
  4. A severity rating: "low", "medium", "high", or "critical" (how much this failure would matter if it \
recurred at scale in production).

You are proposing a candidate for HUMAN REVIEW. Your suggestion is not final and will not be treated as \
ground truth. Be honest about uncertainty rather than confidently wrong.

Respond with ONLY a JSON object, no other text, in this exact shape:
{"failure_category": string, "expected_output": string | null, "difficulty": "easy"|"medium"|"hard", \
"severity": "low"|"medium"|"high"|"critical", "reasoning": string}
"""


class AnthropicClientProtocol(Protocol):
    """Minimal shape we depend on from the anthropic SDK client, so tests
    can supply a lightweight fake without importing the real SDK."""

    def messages_create(self, **kwargs) -> Any:
        ...


class _RealAnthropicClientAdapter:
    """Thin adapter around the actual anthropic.Anthropic() client, so the
    generator's dependency surface (`messages_create`) stays stable even
    if we swap SDK call shapes later."""

    def __init__(self, anthropic_client: Any):
        self._client = anthropic_client

    def messages_create(self, **kwargs) -> Any:
        return self._client.messages.create(**kwargs)


def build_real_client() -> "_RealAnthropicClientAdapter":
    """Constructs a real Anthropic client from environment credentials.
    Only called from the live acceptance suite / production wiring —
    never from the main automated test suite."""
    import anthropic  # imported lazily so the main suite never needs this installed to pass

    return _RealAnthropicClientAdapter(anthropic.Anthropic())


def _build_interaction_prompt(log, signal_types: list[str]) -> str:
    return json.dumps(
        {
            "user_input": log.user_input,
            "assistant_output": log.assistant_output,
            "attached_signal_types": signal_types,
            "status": log.status,
            "retry_of": log.retry_of,
        },
        indent=2,
    )


def _extract_text(response: Any) -> str:
    # Supports both the real anthropic SDK response shape and simple fakes
    # used in tests that just set `.content = [SimpleNamespace(text=...)]`
    blocks = getattr(response, "content", None) or []
    texts = [getattr(b, "text", None) for b in blocks if getattr(b, "text", None)]
    return "\n".join(texts)


_FENCED_JSON_PATTERN = re.compile(r"```(?:json)?\s*\n?(.*?)\n?```", re.DOTALL)


def _strip_markdown_json_fence(text: str) -> str:
    """Models commonly wrap JSON output in ```json ... ``` fences even when
    explicitly instructed to return raw JSON only — this is normal, expected
    model behavior, not a malformed response. Strip the fence if present;
    otherwise return the text unchanged so a genuinely bare JSON response
    (or genuinely malformed output, which should still fail to parse and
    be honestly skipped) isn't affected."""
    stripped = text.strip()
    match = _FENCED_JSON_PATTERN.search(stripped)
    if match:
        return match.group(1).strip()
    return stripped


def _normalize_optional_str(value: Any) -> str | None:
    """Collapse None, empty string, and whitespace-only strings all to
    None, and guard against a model returning a non-string JSON type
    (number/bool/list) for a field that should be text. Prevents an
    empty-string proposal from silently bypassing the "cannot approve
    without grounding" gate, which checks for None specifically."""
    if value is None or not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None


class ClaudeCandidateGenerator(CandidateGenerator):
    def __init__(self, client: AnthropicClientProtocol, model: str = MODEL_NAME):
        self._client = client
        self._model = model

    def generate(self, context: ClusterContext) -> list[EvalCandidate]:
        candidates: list[EvalCandidate] = []
        now = datetime.now(timezone.utc)

        for log in context.representative_logs[:MAX_REPRESENTATIVES_PER_CLUSTER]:
            signal_types = [s.signal_type for s in context.signals_by_interaction.get(log.interaction_id, [])]
            if not signal_types:
                continue

            try:
                response = self._client.messages_create(
                    model=self._model,
                    max_tokens=600,
                    system=_SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": _build_interaction_prompt(log, signal_types)}],
                )
                raw_text = _extract_text(response)
                parsed = json.loads(_strip_markdown_json_fence(raw_text))

                failure_category = _normalize_optional_str(parsed.get("failure_category"))
                expected_output = _normalize_optional_str(parsed.get("expected_output"))
                difficulty: Difficulty = parsed.get("difficulty") if parsed.get("difficulty") in ("easy", "medium", "hard") else "medium"
                severity: Severity = parsed.get("severity") if parsed.get("severity") in ("low", "medium", "high", "critical") else "medium"

                if not failure_category:
                    logger.warning("Claude generator: missing failure_category for %s, skipping", log.interaction_id)
                    continue

            except (json.JSONDecodeError, KeyError, AttributeError, TypeError) as e:
                logger.warning("Claude generator: unparseable response for %s (%s), skipping rather than fabricating", log.interaction_id, e)
                continue
            except Exception as e:  # network/API errors etc — never crash the whole batch over one call
                logger.warning("Claude generator: API call failed for %s (%s), skipping", log.interaction_id, e)
                continue

            candidate = EvalCandidate(
                candidate_id=str(uuid4()),
                source_interaction_ids=[log.interaction_id],
                input_text=log.user_input,
                context=log.system_prompt,
                observed_output=log.assistant_output,
                proposed_expected_output=expected_output,
                proposed_failure_category=failure_category,
                generation_source="model_generated",
                difficulty=difficulty,
                severity=severity,
                cluster_id=context.cluster_id,
                review_status="pending",
                created_at=now,
            )
            candidates.append(candidate)

        self._assert_generator_contract(candidates)
        return candidates
