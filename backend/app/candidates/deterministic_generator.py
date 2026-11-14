"""
Deterministic, rule-based candidate generator.

Used for the main automated test suite and as an offline fallback — it
requires no network access or API credentials, so CI and local
development never depend on a paid service. It derives
proposed_failure_category and proposed_expected_output from the
deterministic/heuristic SIGNALS attached to an interaction, and from any
user_correction present (the strongest available signal of what the
"right" answer should have been, since a real user supplied it).

Its outputs are honest about their limits: generation_source is
"user_correction" when a correction is available, otherwise
"deterministic" (derived purely from signal type -> category mapping) or
"heuristic" when only heuristic signals are present. It never claims to
be "model_generated" because no model was involved.
"""
from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from app.candidates.interface import CandidateGenerator, ClusterContext
from app.schemas.candidate import EvalCandidate

_SIGNAL_TO_CATEGORY = {
    "thumbs_down": "user_rated_negative",
    "http_or_api_error": "system_error",
    "malformed_json_output": "malformed_structured_output",
    "timeout": "latency_or_timeout",
    "retry_occurred": "required_retry",
    "user_correction_present": "user_corrected_response",
    "response_edited": "user_edited_response",
    "large_divergence_after_retry": "inconsistent_retry_behavior",
    "latency_outlier": "latency_or_timeout",
    "refusal_mismatch_suspected": "suspected_incorrect_refusal",
    "unsupported_claim_suspected": "suspected_unsupported_claim",
}

_SEVERITY_BY_CATEGORY = {
    "system_error": "high",
    "malformed_structured_output": "high",
    "suspected_incorrect_refusal": "medium",
    "suspected_unsupported_claim": "medium",
    "user_corrected_response": "medium",
    "inconsistent_retry_behavior": "medium",
    "latency_or_timeout": "low",
    "user_rated_negative": "medium",
    "user_edited_response": "medium",
    "required_retry": "low",
}


_CONTENT_WRONGNESS_SIGNALS = {
    "thumbs_down", "http_or_api_error", "malformed_json_output",
    "user_correction_present", "response_edited", "large_divergence_after_retry",
    "refusal_mismatch_suspected", "unsupported_claim_suspected",
}
_OPERATIONAL_ONLY_SIGNALS = {"timeout", "latency_outlier", "retry_occurred"}


def _pick_category(signal_types: list[str]) -> str:
    for st in signal_types:
        if st in _SIGNAL_TO_CATEGORY:
            return _SIGNAL_TO_CATEGORY[st]
    return "uncategorized_failure"


class DeterministicCandidateGenerator(CandidateGenerator):
    def generate(self, context: ClusterContext) -> list[EvalCandidate]:
        candidates: list[EvalCandidate] = []
        now = datetime.now(timezone.utc)

        for log in context.representative_logs:
            signal_types = [s.signal_type for s in context.signals_by_interaction.get(log.interaction_id, [])]
            if not signal_types:
                continue

            category = _pick_category(signal_types)
            severity = _SEVERITY_BY_CATEGORY.get(category, "medium")

            if log.user_correction:
                proposed_expected = log.user_correction
                generation_source = "user_correction"
            elif log.edited_response:
                proposed_expected = log.edited_response
                generation_source = "user_correction"
            elif (
                log.status == "success"
                and log.assistant_output
                and set(signal_types) <= _OPERATIONAL_ONLY_SIGNALS
            ):
                # The ONLY signals attached are operational (slow, retried,
                # eventually timed out then succeeded) and none of them
                # question the content itself. In that case the delivered
                # content IS a legitimate expected output — the failure
                # being tracked is about latency/retry cost, not
                # correctness. This is grounded fact (what was actually
                # returned and not flagged as wrong), not a fabricated
                # "better" answer, so it's honest to propose it.
                proposed_expected = log.assistant_output
                generation_source = "deterministic"
            else:
                proposed_expected = None
                generation_source = "deterministic" if any(
                    s in ("thumbs_down", "http_or_api_error", "malformed_json_output", "timeout",
                          "retry_occurred", "user_correction_present", "response_edited")
                    for s in signal_types
                ) else "heuristic"

            candidate = EvalCandidate(
                candidate_id=str(uuid4()),
                source_interaction_ids=[log.interaction_id],
                input_text=log.user_input,
                context=log.system_prompt,
                observed_output=log.assistant_output,
                proposed_expected_output=proposed_expected,
                proposed_failure_category=category,
                generation_source=generation_source,  # type: ignore[arg-type]
                difficulty="medium",
                severity=severity,  # type: ignore[arg-type]
                cluster_id=context.cluster_id,
                review_status="pending",
                created_at=now,
            )
            candidates.append(candidate)

        self._assert_generator_contract(candidates)
        return candidates
