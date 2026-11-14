"""
Persistence layer.

V1 design choice: each table stores its canonical object as a JSON blob
(the full Pydantic model, serialized) plus a handful of indexed columns
used for filtering/joins. This avoids hand-maintaining a fully normalized
relational schema in lockstep with the Pydantic models while still giving
real database persistence, querying, and durability across restarts —
appropriate for a V1 that explicitly should not be over-engineered.

Postgres in Docker/production (DATABASE_URL env var), SQLite for local
dev and the automated test suite (default, in-memory or file-based) —
same schema works on both since we lean on JSON columns rather than
Postgres-specific JSONB features.
"""
from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import JSON, Column, DateTime, Float, Integer, String
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass


class InteractionRecord(Base):
    __tablename__ = "interactions"

    interaction_id = Column(String, primary_key=True)
    conversation_id = Column(String, nullable=True, index=True)
    application = Column(String, nullable=False, index=True)
    model_name = Column(String, nullable=False)
    status = Column(String, nullable=False, index=True)
    content_hash = Column(String, nullable=True, index=True)
    timestamp = Column(DateTime, nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    payload = Column(JSON, nullable=False)  # full InteractionLog.model_dump()


class QuarantineRecord(Base):
    __tablename__ = "quarantined_records"

    quarantine_id = Column(String, primary_key=True)
    reason = Column(String, nullable=False, index=True)
    application = Column(String, nullable=True)
    original_interaction_id = Column(String, nullable=True)
    detected_at = Column(DateTime, nullable=False)
    error_detail = Column(String, nullable=False)


class SignalRecord(Base):
    __tablename__ = "signals"

    signal_id = Column(String, primary_key=True)
    interaction_id = Column(String, nullable=False, index=True)
    signal_type = Column(String, nullable=False, index=True)
    source = Column(String, nullable=False, index=True)
    score = Column(Float, nullable=True)
    created_at = Column(DateTime, nullable=False)
    payload = Column(JSON, nullable=False)  # full SignalEvidence.model_dump()


class ClusterAssignmentRecord(Base):
    __tablename__ = "cluster_assignments"

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(String, nullable=False, index=True)
    interaction_id = Column(String, nullable=False, index=True)
    cluster_id = Column(String, nullable=False, index=True)
    probability = Column(Float, nullable=True)


class CandidateRecord(Base):
    __tablename__ = "candidates"

    candidate_id = Column(String, primary_key=True)
    cluster_id = Column(String, nullable=True, index=True)
    review_status = Column(String, nullable=False, index=True)
    generation_source = Column(String, nullable=False, index=True)
    created_at = Column(DateTime, nullable=False)
    payload = Column(JSON, nullable=False)  # full EvalCandidate.model_dump()


class DatasetVersionRecord(Base):
    __tablename__ = "dataset_versions"

    version = Column(Integer, primary_key=True)
    created_at = Column(DateTime, nullable=False)
    case_ids = Column(JSON, nullable=False)
    category_distribution = Column(JSON, nullable=False)
    provenance_distribution = Column(JSON, nullable=False)


class RegressionRunRecord(Base):
    __tablename__ = "regression_runs"

    run_id = Column(String, primary_key=True)
    dataset_version = Column(Integer, nullable=False, index=True)
    created_at = Column(DateTime, nullable=False)
    pass_rate = Column(Float, nullable=False)
    results = Column(JSON, nullable=False)  # list[CaseResult]-shaped dicts
