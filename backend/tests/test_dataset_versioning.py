import json
from datetime import datetime, timezone

import pytest

from app.dataset.versioning import DatasetVersioningError, EvalDataset
from app.review.service import approve_candidate
from app.schemas.candidate import EvalCandidate


def _candidate(cid, **overrides):
    base = dict(
        candidate_id=cid,
        source_interaction_ids=["i1"],
        input_text="What is my balance?",
        observed_output="$42.10",
        proposed_expected_output="$124.50",
        proposed_failure_category="user_corrected_response",
        generation_source="user_correction",
        created_at=datetime.now(timezone.utc),
    )
    base.update(overrides)
    return EvalCandidate(**base)


def test_cannot_create_version_from_pending_candidates():
    dataset = EvalDataset()
    with pytest.raises(DatasetVersioningError):
        dataset.create_version([_candidate("c1")])  # still pending, not approved


def test_create_version_from_approved_candidates():
    dataset = EvalDataset()
    approved = approve_candidate(_candidate("c1"))
    version = dataset.create_version([approved])
    assert version.version == 1
    assert version.approved_case_count == 1
    assert "user_corrected_response" in version.category_distribution


def test_versions_are_immutable_snapshots():
    dataset = EvalDataset()
    c1 = approve_candidate(_candidate("c1"))
    v1 = dataset.create_version([c1])

    c2 = approve_candidate(_candidate("c2"))
    v2 = dataset.create_version([c1, c2])  # v2 is a full snapshot including c1 + new c2

    assert v1.case_ids == ("c1",)
    assert v2.case_ids == ("c1", "c2")
    assert v1.version == 1
    assert v2.version == 2
    # v1 must remain unchanged after v2 is created
    assert dataset.versions[0].case_ids == ("c1",)


def test_version_numbers_increment_sequentially():
    dataset = EvalDataset()
    c1 = approve_candidate(_candidate("c1"))
    dataset.create_version([c1])
    c2 = approve_candidate(_candidate("c2"))
    dataset.create_version([c1, c2])
    c3 = approve_candidate(_candidate("c3"))
    v3 = dataset.create_version([c1, c2, c3])
    assert v3.version == 3


def test_export_jsonl_writes_only_the_specified_version(tmp_path):
    dataset = EvalDataset()
    c1 = approve_candidate(_candidate("c1"))
    dataset.create_version([c1])
    c2 = approve_candidate(_candidate("c2"))
    dataset.create_version([c1, c2])

    out_path = dataset.export_jsonl(1, tmp_path / "v1.jsonl")
    lines = out_path.read_text().strip().split("\n")
    assert len(lines) == 1
    exported = json.loads(lines[0])
    assert exported["candidate_id"] == "c1"


def test_category_distribution_uses_reviewer_category_not_proposed():
    dataset = EvalDataset()
    approved = approve_candidate(
        _candidate("c1", proposed_failure_category="original_guess"),
        reviewer_failure_category="corrected_category",
    )
    version = dataset.create_version([approved])
    assert "corrected_category" in version.category_distribution
    assert "original_guess" not in version.category_distribution


def test_provenance_distribution_tracks_generation_source():
    dataset = EvalDataset()
    c1 = approve_candidate(_candidate("c1", generation_source="user_correction"))
    c2 = approve_candidate(_candidate("c2", generation_source="model_generated"))
    version = dataset.create_version([c1, c2])
    assert version.provenance_distribution == {"user_correction": 1, "model_generated": 1}
