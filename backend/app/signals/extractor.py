"""
Failure signal extraction.

Per design_lock.md section 3, every signal carries an explicit `source`:
  - deterministic: derivable with certainty from structured fields alone
    (thumbs down, http/api error, retry occurred, user edited answer,
    malformed JSON output, timeout).
  - heuristic: a rule-of-thumb over statistical/textual properties that
    can be wrong (latency outlier, large divergence after retry, refusal
    mismatch suspected, unsupported claim suspected).

Neither category is ground truth. Both are prioritization signals that
feed clustering and candidate generation — see design_lock.md's final
principle. model_generated and human_approved signals are produced by
other modules (candidates/, review/), not here.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from difflib import SequenceMatcher
from statistics import mean, pstdev
from uuid import uuid4

from app.schemas.interaction import InteractionLog
from app.schemas.signal import SignalEvidence

# --- heuristic thresholds (documented, tunable, not claimed as precise) ---
LATENCY_OUTLIER_Z_SCORE = 2.5
LATENCY_MIN_SAMPLE_SIZE = 5  # below this, we don't have enough data to call something an outlier
RETRY_DIVERGENCE_THRESHOLD = 0.35  # similarity ratio below this after a retry = "large divergence"

_REFUSAL_PATTERNS = re.compile(
    r"\b(i can'?t help with that|i'?m not able to|i cannot assist|"
    r"i won'?t be able to|as an ai( language model)?, i (can'?t|cannot))\b",
    re.IGNORECASE,
)

_HEDGE_PATTERNS = re.compile(
    r"\b(according to (my|the) (knowledge|training)|i believe|i think|"
    r"as far as i know|it is (likely|possible) that)\b",
    re.IGNORECASE,
)
_OVERCONFIDENT_ABSOLUTE = re.compile(
    r"\b(definitely|certainly|100%|guaranteed|always|never fails|proven fact)\b",
    re.IGNORECASE,
)


def _new_signal(interaction_id: str, signal_type: str, source: str, score, evidence: list[str]) -> SignalEvidence:
    return SignalEvidence(
        signal_id=str(uuid4()),
        interaction_id=interaction_id,
        signal_type=signal_type,  # type: ignore[arg-type]
        source=source,  # type: ignore[arg-type]
        score=score,
        evidence=evidence,
        created_at=datetime.now(timezone.utc),
    )


def _deterministic_signals(log: InteractionLog) -> list[SignalEvidence]:
    signals: list[SignalEvidence] = []

    if log.user_feedback == "negative":
        signals.append(_new_signal(log.interaction_id, "thumbs_down", "deterministic", None, ["user_feedback == negative"]))

    if log.status == "error":
        signals.append(_new_signal(
            log.interaction_id, "http_or_api_error", "deterministic", None,
            [f"status == error (error_type={log.error_type})"],
        ))

    if log.status == "timeout":
        signals.append(_new_signal(log.interaction_id, "timeout", "deterministic", None, ["status == timeout"]))

    if log.status == "malformed_output" or log.structured_output_valid is False:
        signals.append(_new_signal(
            log.interaction_id, "malformed_json_output", "deterministic", None,
            ["structured_output_valid is False" if log.structured_output_valid is False else "status == malformed_output"],
        ))

    if log.retry_of is not None or log.retry_count > 0:
        signals.append(_new_signal(
            log.interaction_id, "retry_occurred", "deterministic", None,
            [f"retry_of={log.retry_of}, retry_count={log.retry_count}"],
        ))

    if log.user_correction:
        signals.append(_new_signal(log.interaction_id, "user_correction_present", "deterministic", None, ["user_correction is set"]))

    if log.edited_response:
        signals.append(_new_signal(log.interaction_id, "response_edited", "deterministic", None, ["edited_response is set"]))

    return signals


def _refusal_mismatch_heuristic(log: InteractionLog) -> list[SignalEvidence]:
    """Flag when the assistant appears to refuse but the interaction shows
    no independent deterministic reason (error/timeout) that would justify
    a refusal — i.e. a plain informational question got refused. This is a
    heuristic, not a determination that the refusal was wrong."""
    if not log.assistant_output or log.status not in ("success", "refusal"):
        return []
    if _REFUSAL_PATTERNS.search(log.assistant_output):
        score = 0.6 if log.status == "success" else 0.4
        return [_new_signal(
            log.interaction_id, "refusal_mismatch_suspected", "heuristic", score,
            ["assistant_output matches refusal language pattern"],
        )]
    return []


def _unsupported_claim_heuristic(log: InteractionLog) -> list[SignalEvidence]:
    """Very coarse heuristic: overconfident absolute language with no
    hedging in a longer response is treated as a weak signal worth
    reviewing, never as proof of hallucination."""
    if not log.assistant_output or len(log.assistant_output) < 40:
        return []
    has_absolute = bool(_OVERCONFIDENT_ABSOLUTE.search(log.assistant_output))
    has_hedge = bool(_HEDGE_PATTERNS.search(log.assistant_output))
    if has_absolute and not has_hedge:
        return [_new_signal(
            log.interaction_id, "unsupported_claim_suspected", "heuristic", 0.3,
            ["overconfident absolute language with no hedging detected"],
        )]
    return []


def _retry_divergence_heuristic(logs_by_id: dict[str, InteractionLog]) -> list[SignalEvidence]:
    """For any interaction that is a retry of another, compare assistant
    outputs; a large divergence is a heuristic worth surfacing (the model
    gave a substantially different answer on retry, which is itself
    interesting regardless of which answer was right)."""
    signals: list[SignalEvidence] = []
    for log in logs_by_id.values():
        if not log.retry_of:
            continue
        original = logs_by_id.get(log.retry_of)
        if not original or not original.assistant_output or not log.assistant_output:
            continue
        ratio = SequenceMatcher(None, original.assistant_output, log.assistant_output).ratio()
        if ratio < RETRY_DIVERGENCE_THRESHOLD:
            signals.append(_new_signal(
                log.interaction_id, "large_divergence_after_retry", "heuristic", round(1 - ratio, 2),
                [f"similarity_ratio={ratio:.2f} vs original interaction {original.interaction_id}"],
            ))
    return signals


def _latency_outlier_heuristic(logs: list[InteractionLog]) -> list[SignalEvidence]:
    """Population-relative z-score per (application, model_name) group.
    Requires a minimum sample size to avoid calling a single slow request
    in a tiny batch an 'outlier' with no statistical basis."""
    signals: list[SignalEvidence] = []
    groups: dict[tuple[str, str], list[InteractionLog]] = {}
    for log in logs:
        if log.latency_ms is None:
            continue
        groups.setdefault((log.application, log.model_name), []).append(log)

    for _, group in groups.items():
        if len(group) < LATENCY_MIN_SAMPLE_SIZE:
            continue
        values = [g.latency_ms for g in group]
        mu = mean(values)
        sigma = pstdev(values)
        if sigma == 0:
            continue
        for log in group:
            z = (log.latency_ms - mu) / sigma
            if z >= LATENCY_OUTLIER_Z_SCORE:
                signals.append(_new_signal(
                    log.interaction_id, "latency_outlier", "heuristic", round(min(z / 5, 1.0), 2),
                    [f"latency_ms={log.latency_ms:.0f}, group_mean={mu:.0f}, z={z:.2f}"],
                ))
    return signals


@dataclass
class SignalExtractionResult:
    signals: list[SignalEvidence]


def extract_signals(logs: list[InteractionLog]) -> SignalExtractionResult:
    """Run all deterministic + heuristic extractors over a batch of
    normalized interactions. Batch-level heuristics (latency outliers,
    retry divergence) need the whole batch as context; per-record
    heuristics/deterministic checks run independently."""
    logs_by_id = {log.interaction_id: log for log in logs}
    signals: list[SignalEvidence] = []

    for log in logs:
        signals.extend(_deterministic_signals(log))
        signals.extend(_refusal_mismatch_heuristic(log))
        signals.extend(_unsupported_claim_heuristic(log))

    signals.extend(_retry_divergence_heuristic(logs_by_id))
    signals.extend(_latency_outlier_heuristic(logs))

    return SignalExtractionResult(signals=signals)
