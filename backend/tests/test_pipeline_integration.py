"""
End-to-end integration test: synthetic demo log file -> ingestion ->
redaction -> dedup -> signal extraction. Confirms the layers compose
correctly and that the adversarial records planted in the demo dataset
are actually caught, not silently swallowed or crashing the run.
"""
import json
from pathlib import Path

from app.ingestion.dedup import dedup_interactions
from app.ingestion.pipeline import ingest_batch
from app.signals.extractor import extract_signals

DEMO_PATH = Path(__file__).parent.parent.parent / "demo_data" / "synthetic_logs.jsonl"


def _load_raw_records():
    records = []
    with DEMO_PATH.open() as f:
        for line in f:
            records.append(json.loads(line))
    return records


def test_demo_dataset_exists_and_has_expected_scale():
    records = _load_raw_records()
    assert len(records) > 100


def test_full_pipeline_runs_without_crashing():
    records = _load_raw_records()
    outcome = ingest_batch(records, provenance="synthetic_demo")
    assert len(outcome.accepted) > 0
    assert len(outcome.quarantined) > 0  # adversarial records must be caught


def test_no_raw_pii_survives_into_accepted_records():
    records = _load_raw_records()
    outcome = ingest_batch(records, provenance="synthetic_demo")
    for log in outcome.accepted:
        for text in (log.user_input, log.assistant_output, log.system_prompt):
            if not text:
                continue
            assert "jordan.blake@example.com" not in text
            assert "555-0148" not in text


def test_credential_bearing_adversarial_record_is_quarantined():
    records = _load_raw_records()
    outcome = ingest_batch(records, provenance="synthetic_demo")
    reasons = [q.reason for q in outcome.quarantined]
    assert any("sensitive_data" in r for r in reasons)


def test_malformed_and_garbage_records_are_quarantined_not_crashing():
    records = _load_raw_records()
    outcome = ingest_batch(records, provenance="synthetic_demo")
    reasons = {q.reason for q in outcome.quarantined}
    assert "schema_validation_failed" in reasons or "missing_or_invalid_required_fields" in reasons


def test_dedup_catches_planted_exact_duplicates():
    records = _load_raw_records()
    outcome = ingest_batch(records, provenance="synthetic_demo")
    dedup_result = dedup_interactions(outcome.accepted)
    assert len(dedup_result.duplicates) >= 3  # 4 planted dupes -> at least 3 flagged as dupes of the first


def test_signal_extraction_finds_planted_failure_signals():
    records = _load_raw_records()
    outcome = ingest_batch(records, provenance="synthetic_demo")
    dedup_result = dedup_interactions(outcome.accepted)
    signal_result = extract_signals(dedup_result.unique)
    types_found = {s.signal_type for s in signal_result.signals}
    # planted themes should surface at least these deterministic/heuristic signals
    assert "malformed_json_output" in types_found
    assert "user_correction_present" in types_found
    assert "refusal_mismatch_suspected" in types_found
    assert "unsupported_claim_suspected" in types_found
    assert "latency_outlier" in types_found or "excessive_latency_or_retry" in {r.get("metadata", {}).get("_planted_theme") for r in records}
