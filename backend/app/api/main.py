"""
FastAPI application.

V1 keeps request handling synchronous and in-process: an ingest call runs
ingestion -> redaction -> dedup -> signal extraction -> clustering ->
candidate generation immediately and persists results. This is
appropriate at demo/V1 scale; a background job queue is a reasonable V2
upgrade, not a V1 requirement (per "don't over-engineer").
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Optional
from uuid import uuid4

from fastapi import Body, Depends, FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.candidates.deterministic_generator import DeterministicCandidateGenerator
from app.candidates.orchestrator import generate_candidates_for_clusters
from app.candidates.quality import evaluate_candidate_quality
from app.clustering.cluster import cluster_flagged_interactions
from app.clustering.cluster_quality import evaluate_cluster_quality
from app.dataset.versioning import DatasetVersioningError, EvalDataset
from app.db import repository as repo
from app.db.session import get_session, init_db
from app.ingestion.dedup import dedup_interactions
from app.ingestion.pipeline import ingest_batch
from app.regression.runner import SimilarityJudge, compare_runs, run_regression
from app.review.service import ReviewError, approve_candidate, reject_candidate

from contextlib import asynccontextmanager


@asynccontextmanager
async def _lifespan(app: FastAPI):
    init_db()
    _sync_dataset_state_from_db()
    yield


app = FastAPI(title="Eval Forge — Automated Eval Dataset Generator", version="0.1.0", lifespan=_lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # V1 demo scope; tighten for real deployment
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_db():
    db = get_session()
    try:
        yield db
    finally:
        db.close()


def _sync_dataset_state_from_db():
    """Reconstruct in-memory EvalDataset version-number/case-id state from
    the DB on startup, so a server restart doesn't collide with or lose
    track of previously created (immutable) versions."""
    from app.dataset.versioning import DatasetVersion

    db = get_session()
    try:
        persisted_versions = repo.load_dataset_versions(db)
        all_candidates = {c.candidate_id: c for c in repo.load_candidates(db)}
        for v in persisted_versions:
            for cid in v["case_ids"]:
                if cid in all_candidates:
                    _dataset._cases_by_id[cid] = all_candidates[cid]
            _dataset._versions.append(DatasetVersion(
                version=v["version"],
                created_at=datetime.fromisoformat(v["created_at"]),
                case_ids=tuple(v["case_ids"]),
                category_distribution=v["category_distribution"],
                provenance_distribution=v["provenance_distribution"],
            ))
    finally:
        db.close()


# ---- ingestion -----------------------------------------------------------

class IngestResponse(BaseModel):
    accepted_count: int
    quarantined_count: int
    deduped_count: int
    unique_count: int
    new_signals_found: int
    clusters_found: int
    new_candidates_generated: int


@app.post("/ingest", response_model=IngestResponse)
async def ingest_logs(file: UploadFile, db: Session = Depends(get_db)):
    raw_bytes = await file.read()
    text = raw_bytes.decode("utf-8", errors="replace")

    records = []
    if file.filename and file.filename.endswith(".jsonl"):
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                records.append({"_unparseable_line": line[:200]})
    else:
        try:
            parsed = json.loads(text)
            records = parsed if isinstance(parsed, list) else [parsed]
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="File is not valid JSON or JSONL.")

    outcome = ingest_batch(records, provenance="raw_log")
    repo.save_interactions(db, outcome.accepted)
    repo.save_quarantined(db, outcome.quarantined)

    all_logs = repo.load_interactions(db)
    dedup_result = dedup_interactions(all_logs)

    from app.signals.extractor import extract_signals
    signal_result = extract_signals(dedup_result.unique)
    repo.save_signals(db, signal_result.signals)
    all_signals = repo.load_signals(db)

    cluster_result = cluster_flagged_interactions(dedup_result.unique, all_signals)
    run_id = str(uuid4())
    repo.save_cluster_assignments(db, run_id, cluster_result.assignments)

    new_candidates = generate_candidates_for_clusters(
        DeterministicCandidateGenerator(), cluster_result.assignments,
        dedup_result.unique, all_signals, cluster_result.cluster_ids,
    )
    already_covered = {
        iid for c in repo.load_candidates(db) for iid in c.source_interaction_ids
    }
    new_candidates = [c for c in new_candidates if not set(c.source_interaction_ids) & already_covered]
    repo.save_candidates(db, new_candidates)

    return IngestResponse(
        accepted_count=len(outcome.accepted),
        quarantined_count=len(outcome.quarantined),
        deduped_count=len(dedup_result.duplicates),
        unique_count=len(dedup_result.unique),
        new_signals_found=len(signal_result.signals),
        clusters_found=len(cluster_result.cluster_ids),
        new_candidates_generated=len(new_candidates),
    )


# ---- clusters --------------------------------------------------------

@app.get("/clusters")
def list_clusters(db: Session = Depends(get_db)):
    candidates = repo.load_candidates(db)

    clusters: dict[str, dict] = {}
    for c in candidates:
        if not c.cluster_id:
            continue
        entry = clusters.setdefault(c.cluster_id, {"cluster_id": c.cluster_id, "candidate_count": 0, "examples": []})
        entry["candidate_count"] += 1
        if len(entry["examples"]) < 3:
            entry["examples"].append({"input_text": c.input_text, "observed_output": c.observed_output, "category": c.proposed_failure_category})

    return {"clusters": sorted(clusters.values(), key=lambda c: -c["candidate_count"])}


# ---- candidates / review --------------------------------------------------

@app.get("/candidates")
def list_candidates(status: Optional[str] = None, db: Session = Depends(get_db)):
    candidates = repo.load_candidates(db, review_status=status)
    return {"candidates": [c.model_dump(mode="json") for c in candidates]}


class ReviewRequest(BaseModel):
    reviewer_expected_output: Optional[str] = None
    reviewer_failure_category: Optional[str] = None
    reviewer_notes: Optional[str] = None


@app.post("/candidates/{candidate_id}/approve")
def approve(candidate_id: str, body: ReviewRequest = Body(default=ReviewRequest()), db: Session = Depends(get_db)):
    candidate = repo.get_candidate(db, candidate_id)
    if not candidate:
        raise HTTPException(status_code=404, detail="Candidate not found")
    try:
        approved = approve_candidate(
            candidate,
            reviewer_expected_output=body.reviewer_expected_output,
            reviewer_failure_category=body.reviewer_failure_category,
            reviewer_notes=body.reviewer_notes,
        )
    except ReviewError as e:
        raise HTTPException(status_code=400, detail=str(e))
    repo.save_candidates(db, [approved])
    return approved.model_dump(mode="json")


@app.post("/candidates/{candidate_id}/reject")
def reject(candidate_id: str, body: ReviewRequest = Body(default=ReviewRequest()), db: Session = Depends(get_db)):
    candidate = repo.get_candidate(db, candidate_id)
    if not candidate:
        raise HTTPException(status_code=404, detail="Candidate not found")
    rejected = reject_candidate(candidate, reviewer_notes=body.reviewer_notes)
    repo.save_candidates(db, [rejected])
    return rejected.model_dump(mode="json")


# ---- dataset versioning ----------------------------------------------------

_dataset = EvalDataset()  # process-local; version metadata also persisted to DB for durability


@app.get("/dataset/versions")
def list_dataset_versions(db: Session = Depends(get_db)):
    return {"versions": repo.load_dataset_versions(db)}


@app.post("/dataset/versions")
def create_dataset_version(db: Session = Depends(get_db)):
    approved = repo.load_candidates(db, review_status="approved")
    canonical_ready = [c for c in approved if c.is_canonical_ready()]
    if not canonical_ready:
        raise HTTPException(status_code=400, detail="No canonical-ready approved candidates to version.")
    try:
        version = _dataset.create_version(canonical_ready)
    except DatasetVersioningError as e:
        raise HTTPException(status_code=400, detail=str(e))
    repo.save_dataset_version(db, version)
    return {
        "version": version.version, "approved_case_count": version.approved_case_count,
        "category_distribution": version.category_distribution,
        "provenance_distribution": version.provenance_distribution,
    }


@app.get("/dataset/versions/{version}/export")
def export_dataset_version(version: int, db: Session = Depends(get_db)):
    versions = repo.load_dataset_versions(db)
    match = next((v for v in versions if v["version"] == version), None)
    if not match:
        raise HTTPException(status_code=404, detail="Version not found")
    all_candidates = {c.candidate_id: c for c in repo.load_candidates(db)}
    lines = [all_candidates[cid].model_dump_json() for cid in match["case_ids"] if cid in all_candidates]
    return {"jsonl": "\n".join(lines)}


# ---- regression -----------------------------------------------------------

@app.post("/regression/run")
def trigger_regression_run(dataset_version: Optional[int] = None, db: Session = Depends(get_db)):
    versions = repo.load_dataset_versions(db)
    if not versions:
        raise HTTPException(status_code=400, detail="No dataset versions exist yet.")
    target_version = dataset_version or versions[-1]["version"]
    match = next((v for v in versions if v["version"] == target_version), None)
    if not match:
        raise HTTPException(status_code=404, detail="Version not found")

    all_candidates = {c.candidate_id: c for c in repo.load_candidates(db)}
    cases = [all_candidates[cid] for cid in match["case_ids"] if cid in all_candidates]

    def stub_system_under_test(case):
        # V1 "system under test": a passthrough stub returning the observed
        # (pre-fix) output, i.e. simulating a re-run of the SAME buggy
        # system — should show a low pass rate. Swapping in a real
        # system-under-test callable is a one-line change here.
        return case.observed_output or ""

    run = run_regression(cases, stub_system_under_test, dataset_version=target_version, judge=SimilarityJudge())
    previous_run = repo.load_latest_regression_run(db)
    comparison = compare_runs(run, previous_run)
    repo.save_regression_run(db, run)

    return {
        "run_id": run.run_id,
        "dataset_version": run.dataset_version,
        "pass_rate": run.pass_rate,
        "failures_by_category": run.failures_by_category,
        "comparison": {
            "previous_run_id": comparison.previous_run_id,
            "pass_rate_delta": comparison.pass_rate_delta,
            "regressions": comparison.regressions,
            "improvements": comparison.improvements,
            "new_cases": comparison.new_cases,
        },
    }


@app.get("/regression/runs")
def list_regression_runs(db: Session = Depends(get_db)):
    return {"runs": repo.load_all_regression_runs(db)}


# ---- dashboard --------------------------------------------------------

@app.get("/dashboard/metrics")
def dashboard_metrics(db: Session = Depends(get_db)):
    logs = repo.load_interactions(db)
    quarantined = repo.load_quarantined(db)
    signals = repo.load_signals(db)
    candidates = repo.load_candidates(db)
    versions = repo.load_dataset_versions(db)
    runs = repo.load_all_regression_runs(db)

    signal_type_counts: dict[str, int] = {}
    for s in signals:
        signal_type_counts[s.signal_type] = signal_type_counts.get(s.signal_type, 0) + 1

    candidate_status_counts: dict[str, int] = {}
    for c in candidates:
        candidate_status_counts[c.review_status] = candidate_status_counts.get(c.review_status, 0) + 1

    category_counts: dict[str, int] = {}
    for c in candidates:
        if c.proposed_failure_category:
            category_counts[c.proposed_failure_category] = category_counts.get(c.proposed_failure_category, 0) + 1

    cluster_ids = {c.cluster_id for c in candidates if c.cluster_id}

    return {
        "logs_ingested": len(logs),
        "logs_quarantined": len(quarantined),
        "signals_found": len(signals),
        "signal_type_breakdown": signal_type_counts,
        "clusters_discovered": len(cluster_ids),
        "candidates_generated": len(candidates),
        "candidate_status_breakdown": candidate_status_counts,
        "top_failure_categories": dict(sorted(category_counts.items(), key=lambda kv: -kv[1])[:10]),
        "dataset_versions": len(versions),
        "current_approved_case_count": versions[-1]["approved_case_count"] if versions else 0,
        "latest_regression_pass_rate": runs[0]["pass_rate"] if runs else None,
    }


@app.get("/health")
def health():
    return {"status": "ok", "time": datetime.now(timezone.utc).isoformat()}
