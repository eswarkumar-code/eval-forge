"""
API-level integration tests. Each test gets a fresh SQLite DB file
(via monkeypatched DATABASE_URL + app re-import) so tests don't leak
state into each other or depend on ordering.
"""
import importlib
import json
import os
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

DEMO_PATH = Path(__file__).parent.parent.parent / "demo_data" / "synthetic_logs.jsonl"


@pytest.fixture()
def client(tmp_path, monkeypatch):
    db_path = tmp_path / "test.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")

    # fresh import of session/repository/main so the new DATABASE_URL takes effect
    for mod_name in list(sys.modules):
        if mod_name.startswith("app.db") or mod_name == "app.api.main":
            del sys.modules[mod_name]

    main_module = importlib.import_module("app.api.main")
    with TestClient(main_module.app) as c:
        yield c


def _upload_demo_file(client):
    content = DEMO_PATH.read_bytes()
    return client.post("/ingest", files={"file": ("synthetic_logs.jsonl", content, "application/json")})


def test_health_check(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_ingest_demo_dataset(client):
    resp = _upload_demo_file(client)
    assert resp.status_code == 200
    body = resp.json()
    assert body["accepted_count"] > 0
    assert body["quarantined_count"] > 0
    assert body["clusters_found"] > 0
    assert body["new_candidates_generated"] > 0


def test_ingest_malformed_json_file_returns_400(client):
    resp = client.post("/ingest", files={"file": ("bad.json", b"not json at all {{{", "application/json")})
    assert resp.status_code == 400


def test_dashboard_metrics_after_ingest(client):
    _upload_demo_file(client)
    resp = client.get("/dashboard/metrics")
    assert resp.status_code == 200
    metrics = resp.json()
    assert metrics["logs_ingested"] > 0
    assert metrics["signals_found"] > 0
    assert metrics["clusters_discovered"] > 0
    assert metrics["candidates_generated"] > 0


def test_dashboard_metrics_empty_before_ingest(client):
    resp = client.get("/dashboard/metrics")
    metrics = resp.json()
    assert metrics["logs_ingested"] == 0
    assert metrics["candidates_generated"] == 0


def test_list_candidates_and_review_flow(client):
    _upload_demo_file(client)
    listed = client.get("/candidates", params={"status": "pending"}).json()["candidates"]
    assert len(listed) > 0

    groundable = next((c for c in listed if c["proposed_expected_output"]), None)
    assert groundable is not None, "expected at least one groundable candidate from the demo dataset"

    approve_resp = client.post(f"/candidates/{groundable['candidate_id']}/approve")
    assert approve_resp.status_code == 200
    approved = approve_resp.json()
    assert approved["review_status"] == "approved"
    assert approved["reviewer_expected_output"] is not None


def test_cannot_approve_ungroundable_candidate_without_override(client):
    _upload_demo_file(client)
    listed = client.get("/candidates", params={"status": "pending"}).json()["candidates"]
    ungroundable = next((c for c in listed if not c["proposed_expected_output"]), None)
    if ungroundable is None:
        pytest.skip("no ungroundable candidate in this run")
    resp = client.post(f"/candidates/{ungroundable['candidate_id']}/approve")
    assert resp.status_code == 400


def test_reject_candidate(client):
    _upload_demo_file(client)
    listed = client.get("/candidates", params={"status": "pending"}).json()["candidates"]
    target = listed[0]
    resp = client.post(f"/candidates/{target['candidate_id']}/reject", json={"reviewer_notes": "not useful"})
    assert resp.status_code == 200
    assert resp.json()["review_status"] == "rejected"


def test_dataset_version_creation_requires_approved_candidates(client):
    _upload_demo_file(client)
    resp = client.post("/dataset/versions")
    assert resp.status_code == 400  # nothing approved yet


def test_full_flow_ingest_review_version_export_regression(client):
    _upload_demo_file(client)
    listed = client.get("/candidates", params={"status": "pending"}).json()["candidates"]
    approved_ids = []
    for c in listed:
        if c["proposed_expected_output"]:
            r = client.post(f"/candidates/{c['candidate_id']}/approve")
            assert r.status_code == 200
            approved_ids.append(c["candidate_id"])

    assert len(approved_ids) > 0

    version_resp = client.post("/dataset/versions")
    assert version_resp.status_code == 200
    version = version_resp.json()
    assert version["version"] == 1
    assert version["approved_case_count"] == len(approved_ids)

    export_resp = client.get(f"/dataset/versions/{version['version']}/export")
    assert export_resp.status_code == 200
    lines = export_resp.json()["jsonl"].strip().split("\n")
    assert len(lines) == len(approved_ids)

    regression_resp = client.post("/regression/run")
    assert regression_resp.status_code == 200
    regression = regression_resp.json()
    assert "pass_rate" in regression
    # stub SUT returns the SAME observed (bad) output -> should mismatch
    # the reviewer-approved expected output for most/all cases
    assert regression["pass_rate"] < 1.0

    runs_resp = client.get("/regression/runs")
    assert len(runs_resp.json()["runs"]) == 1

    metrics = client.get("/dashboard/metrics").json()
    assert metrics["dataset_versions"] == 1
    assert metrics["latest_regression_pass_rate"] is not None


def test_second_ingest_does_not_duplicate_candidates_for_same_interactions(client):
    _upload_demo_file(client)
    first_metrics = client.get("/dashboard/metrics").json()
    _upload_demo_file(client)  # re-ingest the exact same file
    second_metrics = client.get("/dashboard/metrics").json()
    # candidate count should not double just because the same file was
    # uploaded twice (dedup + already-covered-interaction guard)
    assert second_metrics["candidates_generated"] <= first_metrics["candidates_generated"] * 1.2
