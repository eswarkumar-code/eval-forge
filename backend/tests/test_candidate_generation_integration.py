import json
from pathlib import Path

from app.candidates.deterministic_generator import DeterministicCandidateGenerator
from app.candidates.orchestrator import generate_candidates_for_clusters
from app.candidates.quality import evaluate_candidate_quality
from app.clustering.cluster import cluster_flagged_interactions
from app.ingestion.dedup import dedup_interactions
from app.ingestion.pipeline import ingest_batch
from app.signals.extractor import extract_signals

DEMO_PATH = Path(__file__).parent.parent.parent / "demo_data" / "synthetic_logs.jsonl"


def _pipeline_through_clustering():
    records = [json.loads(line) for line in DEMO_PATH.open()]
    outcome = ingest_batch(records, provenance="synthetic_demo")
    dedup_result = dedup_interactions(outcome.accepted)
    signal_result = extract_signals(dedup_result.unique)
    cluster_result = cluster_flagged_interactions(dedup_result.unique, signal_result.signals)
    return dedup_result.unique, signal_result.signals, cluster_result


def test_end_to_end_candidate_generation_offline():
    logs, signals, cluster_result = _pipeline_through_clustering()
    candidates = generate_candidates_for_clusters(
        DeterministicCandidateGenerator(), cluster_result.assignments, logs, signals, cluster_result.cluster_ids,
    )
    assert len(candidates) > 10
    assert all(c.review_status == "pending" for c in candidates)
    assert all(c.reviewer_expected_output is None for c in candidates)


def test_generated_candidates_cover_multiple_categories():
    logs, signals, cluster_result = _pipeline_through_clustering()
    candidates = generate_candidates_for_clusters(
        DeterministicCandidateGenerator(), cluster_result.assignments, logs, signals, cluster_result.cluster_ids,
    )
    categories = {c.proposed_failure_category for c in candidates}
    assert len(categories) >= 4


def test_quality_evaluation_runs_over_all_generated_candidates():
    logs, signals, cluster_result = _pipeline_through_clustering()
    candidates = generate_candidates_for_clusters(
        DeterministicCandidateGenerator(), cluster_result.assignments, logs, signals, cluster_result.cluster_ids,
    )
    approved_so_far = []
    clean_count = 0
    flagged_count = 0
    for c in candidates:
        result = evaluate_candidate_quality(c, existing_canonical=approved_so_far)
        if result.is_clean:
            clean_count += 1
        else:
            flagged_count += 1
    assert clean_count > 0
    # some low-severity/no-correction candidates are expected to get flagged
    # (e.g. no observed_output-based grounding) — that's the point of the
    # rubric, not a bug.
    print(f"\ncandidate quality: clean={clean_count} flagged={flagged_count} total={len(candidates)}")
