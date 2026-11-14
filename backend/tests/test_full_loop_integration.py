import json
from pathlib import Path

from app.candidates.deterministic_generator import DeterministicCandidateGenerator
from app.candidates.orchestrator import generate_candidates_for_clusters
from app.clustering.cluster import cluster_flagged_interactions
from app.dataset.versioning import EvalDataset
from app.ingestion.dedup import dedup_interactions
from app.ingestion.pipeline import ingest_batch
from app.review.service import approve_candidate, reject_candidate
from app.signals.extractor import extract_signals

DEMO_PATH = Path(__file__).parent.parent.parent / "demo_data" / "synthetic_logs.jsonl"


def test_full_loop_ingest_through_versioned_export(tmp_path):
    records = [json.loads(line) for line in DEMO_PATH.open()]
    outcome = ingest_batch(records, provenance="synthetic_demo")
    dedup_result = dedup_interactions(outcome.accepted)
    signal_result = extract_signals(dedup_result.unique)
    cluster_result = cluster_flagged_interactions(dedup_result.unique, signal_result.signals)

    candidates = generate_candidates_for_clusters(
        DeterministicCandidateGenerator(), cluster_result.assignments,
        dedup_result.unique, signal_result.signals, cluster_result.cluster_ids,
    )
    assert len(candidates) > 10

    # simulate a human reviewer: approve candidates that have a proposed
    # expected output (groundable), reject the rest
    approved = []
    rejected = []
    for c in candidates:
        if c.proposed_expected_output:
            approved.append(approve_candidate(c))
        else:
            rejected.append(reject_candidate(c, reviewer_notes="insufficient grounding to approve in V1"))

    assert len(approved) > 0
    assert all(c.is_canonical_ready() for c in approved)
    assert all(not c.is_canonical_ready() for c in rejected)

    dataset = EvalDataset()
    version = dataset.create_version(approved)
    assert version.version == 1
    assert version.approved_case_count == len(approved)

    export_path = dataset.export_jsonl(1, tmp_path / "eval_v1.jsonl")
    exported_lines = export_path.read_text().strip().split("\n")
    assert len(exported_lines) == len(approved)

    # sanity: exported cases are genuinely reviewer-sourced, not just the
    # model's raw proposal silently relabeled
    for line in exported_lines:
        case = json.loads(line)
        assert case["review_status"] == "approved"
        assert case["reviewer_expected_output"] is not None
        assert case["reviewer_failure_category"] is not None

    print(f"\nfull loop: {len(candidates)} candidates -> {len(approved)} approved, "
          f"{len(rejected)} rejected -> dataset v{version.version} exported")
