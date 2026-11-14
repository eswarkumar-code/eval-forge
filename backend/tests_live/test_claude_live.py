"""
LIVE ACCEPTANCE SUITE — exercises the real Anthropic API.

This is intentionally NOT part of the main automated test suite
(backend/tests/). It is never imported or collected by a plain
`pytest tests/` run, and it is skipped even when run directly unless
BOTH of the following are true:
  1. RUN_LIVE_CLAUDE_TESTS=1 is set in the environment (explicit opt-in)
  2. ANTHROPIC_API_KEY is present in the environment

Run it explicitly with:
    RUN_LIVE_CLAUDE_TESTS=1 pytest backend/tests_live/ -v -s

It uses ONLY synthetic/already-redacted records from demo_data/ — never
any data from another project, and never raw unredacted PII (the
synthetic dataset's raw PII fields are redacted by the ingestion pipeline
before anything is sent to Claude, exactly as in production).

Purpose: confirm the real ClaudeCandidateGenerator (a) successfully calls
the live API, (b) returns a well-formed response we can parse, (c) still
honors the never-auto-canonical contract with a REAL model in the loop,
not just a mock.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    not (os.environ.get("RUN_LIVE_CLAUDE_TESTS") == "1" and os.environ.get("ANTHROPIC_API_KEY")),
    reason="Live Claude acceptance suite requires RUN_LIVE_CLAUDE_TESTS=1 and ANTHROPIC_API_KEY to be set explicitly.",
)

DEMO_PATH = Path(__file__).parent.parent.parent / "demo_data" / "synthetic_logs.jsonl"


def _load_pipeline_output():
    from app.ingestion.dedup import dedup_interactions
    from app.ingestion.pipeline import ingest_batch
    from app.signals.extractor import extract_signals
    from app.clustering.cluster import cluster_flagged_interactions

    records = [json.loads(line) for line in DEMO_PATH.open()]
    outcome = ingest_batch(records, provenance="synthetic_demo")
    dedup_result = dedup_interactions(outcome.accepted)
    signal_result = extract_signals(dedup_result.unique)
    cluster_result = cluster_flagged_interactions(dedup_result.unique, signal_result.signals)
    return dedup_result.unique, signal_result.signals, cluster_result


def test_live_claude_generator_produces_valid_candidates():
    from app.candidates.claude_generator import ClaudeCandidateGenerator, build_real_client
    from app.candidates.orchestrator import generate_candidates_for_clusters

    logs, signals, cluster_result = _load_pipeline_output()
    # keep the live call small and cheap: just the first 2 clusters
    limited_cluster_ids = cluster_result.cluster_ids[:2]

    client = build_real_client()
    generator = ClaudeCandidateGenerator(client)
    candidates = generate_candidates_for_clusters(
        generator, cluster_result.assignments, logs, signals, limited_cluster_ids,
    )

    assert len(candidates) > 0, "Live API call produced zero candidates — check API key/model access"
    for c in candidates:
        assert c.generation_source == "model_generated"
        assert c.review_status == "pending"
        assert c.reviewer_expected_output is None
        assert c.reviewer_failure_category is None
        assert c.proposed_failure_category is not None
        # never sent raw PII to the model — spot check the input text itself
        assert "@example.com" not in c.input_text
        # explicit type/value check on proposed_expected_output: must be
        # either a genuinely non-empty string or exactly None -- never an
        # empty or whitespace-only string. This is the specific check that
        # would have caught the empty-string-vs-null bug found earlier in
        # Gate 1; do not rely on print formatting to represent this.
        assert c.proposed_expected_output is None or (
            isinstance(c.proposed_expected_output, str) and c.proposed_expected_output.strip() != ""
        ), f"proposed_expected_output must be None or non-blank, got {c.proposed_expected_output!r}"
        print(f"\n[live] category={c.proposed_failure_category!r} severity={c.severity} "
              f"expected_output_repr={c.proposed_expected_output!r}")


def test_live_claude_generator_handles_ambiguous_case_honestly():
    """For a genuinely ambiguous interaction, the model should be able to
    return a null expected_output rather than confidently fabricating one
    — we just check the call completes and produces a well-formed
    candidate either way; we don't assert null specifically since the
    live model's judgment on any one case isn't something we control."""
    from app.candidates.claude_generator import ClaudeCandidateGenerator, build_real_client
    from app.candidates.interface import ClusterContext
    from app.schemas.interaction import InteractionLog
    from app.schemas.signal import SignalEvidence
    from datetime import datetime, timezone
    from uuid import uuid4

    log = InteractionLog(
        interaction_id=str(uuid4()),
        timestamp=datetime.now(timezone.utc),
        application="support-bot",
        model_provider="anthropic",
        model_name="claude-sonnet",
        user_input="It's not working.",
        assistant_output="Could you share more detail about what's happening?",
        status="success",
        provenance="synthetic_demo",
    )
    signal = SignalEvidence(
        signal_id=str(uuid4()), interaction_id=log.interaction_id,
        signal_type="unsupported_claim_suspected", source="heuristic",
        score=0.3, evidence=["ambiguous request"], created_at=datetime.now(timezone.utc),
    )
    ctx = ClusterContext(
        cluster_id="live_test_ambiguous",
        representative_logs=[log],
        signals_by_interaction={log.interaction_id: [signal]},
        dominant_signal_types=["unsupported_claim_suspected"],
    )
    client = build_real_client()
    candidates = ClaudeCandidateGenerator(client).generate(ctx)
    assert len(candidates) <= 1  # either a well-formed candidate or none (honest skip), never a crash
    if candidates:
        assert candidates[0].review_status == "pending"
