from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from sqlalchemy.orm import Session

from app.db.models import (
    CandidateRecord, ClusterAssignmentRecord, DatasetVersionRecord,
    InteractionRecord, QuarantineRecord, RegressionRunRecord, SignalRecord,
)
from app.dataset.versioning import DatasetVersion
from app.ingestion.pipeline import QuarantinedRecord
from app.regression.runner import CaseResult, RegressionRun
from app.schemas.candidate import EvalCandidate
from app.schemas.interaction import InteractionLog
from app.schemas.signal import SignalEvidence


# ---- interactions ------------------------------------------------------

def save_interactions(db: Session, logs: list[InteractionLog]) -> None:
    for log in logs:
        existing = db.get(InteractionRecord, log.interaction_id)
        payload = log.model_dump(mode="json")
        if existing:
            existing.payload = payload
        else:
            db.add(InteractionRecord(
                interaction_id=log.interaction_id,
                conversation_id=log.conversation_id,
                application=log.application,
                model_name=log.model_name,
                status=log.status,
                content_hash=log.content_hash,
                timestamp=log.timestamp,
                payload=payload,
            ))
    db.commit()


def load_interactions(db: Session, application: Optional[str] = None) -> list[InteractionLog]:
    q = db.query(InteractionRecord)
    if application:
        q = q.filter(InteractionRecord.application == application)
    return [InteractionLog(**row.payload) for row in q.all()]


def save_quarantined(db: Session, records: list[QuarantinedRecord]) -> None:
    for r in records:
        db.add(QuarantineRecord(
            quarantine_id=r.quarantine_id,
            reason=r.reason,
            application=r.application,
            original_interaction_id=r.original_interaction_id,
            detected_at=r.detected_at,
            error_detail=r.error_detail,
        ))
    db.commit()


def load_quarantined(db: Session) -> list[dict]:
    return [
        {
            "quarantine_id": r.quarantine_id, "reason": r.reason, "application": r.application,
            "original_interaction_id": r.original_interaction_id,
            "detected_at": r.detected_at.isoformat(), "error_detail": r.error_detail,
        }
        for r in db.query(QuarantineRecord).all()
    ]


# ---- signals -------------------------------------------------------------

def save_signals(db: Session, signals: list[SignalEvidence]) -> None:
    for s in signals:
        if db.get(SignalRecord, s.signal_id):
            continue
        db.add(SignalRecord(
            signal_id=s.signal_id, interaction_id=s.interaction_id, signal_type=s.signal_type,
            source=s.source, score=s.score, created_at=s.created_at, payload=s.model_dump(mode="json"),
        ))
    db.commit()


def load_signals(db: Session) -> list[SignalEvidence]:
    return [SignalEvidence(**row.payload) for row in db.query(SignalRecord).all()]


# ---- clustering ------------------------------------------------------------

def save_cluster_assignments(db: Session, run_id: str, assignments) -> None:
    for a in assignments:
        db.add(ClusterAssignmentRecord(
            run_id=run_id, interaction_id=a.interaction_id, cluster_id=a.cluster_id, probability=a.probability,
        ))
    db.commit()


def load_latest_cluster_assignments(db: Session, run_id: str) -> list[dict]:
    rows = db.query(ClusterAssignmentRecord).filter(ClusterAssignmentRecord.run_id == run_id).all()
    return [{"interaction_id": r.interaction_id, "cluster_id": r.cluster_id, "probability": r.probability} for r in rows]


# ---- candidates --------------------------------------------------------

def save_candidates(db: Session, candidates: list[EvalCandidate]) -> None:
    for c in candidates:
        existing = db.get(CandidateRecord, c.candidate_id)
        payload = c.model_dump(mode="json")
        if existing:
            existing.review_status = c.review_status
            existing.payload = payload
        else:
            db.add(CandidateRecord(
                candidate_id=c.candidate_id, cluster_id=c.cluster_id, review_status=c.review_status,
                generation_source=c.generation_source, created_at=c.created_at, payload=payload,
            ))
    db.commit()


def load_candidates(db: Session, review_status: Optional[str] = None) -> list[EvalCandidate]:
    q = db.query(CandidateRecord)
    if review_status:
        q = q.filter(CandidateRecord.review_status == review_status)
    return [EvalCandidate(**row.payload) for row in q.all()]


def get_candidate(db: Session, candidate_id: str) -> Optional[EvalCandidate]:
    row = db.get(CandidateRecord, candidate_id)
    return EvalCandidate(**row.payload) if row else None


# ---- dataset versions ----------------------------------------------------

def save_dataset_version(db: Session, version: DatasetVersion) -> None:
    db.add(DatasetVersionRecord(
        version=version.version, created_at=version.created_at,
        case_ids=list(version.case_ids),
        category_distribution=version.category_distribution,
        provenance_distribution=version.provenance_distribution,
    ))
    db.commit()


def load_dataset_versions(db: Session) -> list[dict]:
    rows = db.query(DatasetVersionRecord).order_by(DatasetVersionRecord.version).all()
    return [
        {
            "version": r.version, "created_at": r.created_at.isoformat(),
            "case_ids": r.case_ids, "approved_case_count": len(r.case_ids),
            "category_distribution": r.category_distribution,
            "provenance_distribution": r.provenance_distribution,
        }
        for r in rows
    ]


def get_latest_dataset_version_number(db: Session) -> int:
    row = db.query(DatasetVersionRecord).order_by(DatasetVersionRecord.version.desc()).first()
    return row.version if row else 0


# ---- regression runs -----------------------------------------------------

def save_regression_run(db: Session, run: RegressionRun) -> None:
    db.add(RegressionRunRecord(
        run_id=run.run_id, dataset_version=run.dataset_version, created_at=run.created_at,
        pass_rate=run.pass_rate,
        results=[
            {"case_id": r.case_id, "category": r.category, "passed": r.passed, "score": r.score, "actual_output": r.actual_output}
            for r in run.results
        ],
    ))
    db.commit()


def load_regression_run(db: Session, run_id: str) -> Optional[RegressionRun]:
    row = db.get(RegressionRunRecord, run_id)
    if not row:
        return None
    run = RegressionRun(run_id=row.run_id, dataset_version=row.dataset_version, created_at=row.created_at)
    run.results = [CaseResult(**r) for r in row.results]
    return run


def load_latest_regression_run(db: Session) -> Optional[RegressionRun]:
    row = db.query(RegressionRunRecord).order_by(RegressionRunRecord.created_at.desc()).first()
    return load_regression_run(db, row.run_id) if row else None


def load_all_regression_runs(db: Session) -> list[dict]:
    rows = db.query(RegressionRunRecord).order_by(RegressionRunRecord.created_at.desc()).all()
    return [
        {"run_id": r.run_id, "dataset_version": r.dataset_version, "created_at": r.created_at.isoformat(), "pass_rate": r.pass_rate}
        for r in rows
    ]
