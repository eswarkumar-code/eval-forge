from __future__ import annotations

import os

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.db.models import Base

DEFAULT_SQLITE_URL = "sqlite:///./eval_forge.db"


def get_database_url() -> str:
    return os.environ.get("DATABASE_URL", DEFAULT_SQLITE_URL)


def make_engine(database_url: str | None = None):
    url = database_url or get_database_url()
    connect_args = {"check_same_thread": False} if url.startswith("sqlite") else {}
    # pool_pre_ping: test each pooled connection with a lightweight SELECT 1
    # before handing it to a request. Without this, a connection that the
    # DB server silently closed (container restart, load balancer idle
    # timeout, network blip) raises OperationalError on first use instead
    # of being transparently replaced. This is the standard mitigation for
    # "server closed the connection unexpectedly" in containerized Postgres
    # deployments — see https://docs.sqlalchemy.org/en/20/core/pooling.html#disconnect-handling-pessimistic
    # pool_recycle: proactively recycle connections older than 280s so we
    # never hit a server-side idle-connection timeout in the first place.
    # Only meaningful for real DB backends; SQLite doesn't pool the same way.
    pool_kwargs = {} if url.startswith("sqlite") else {"pool_pre_ping": True, "pool_recycle": 280}
    return create_engine(url, connect_args=connect_args, **pool_kwargs)


_engine = make_engine()
SessionLocal = sessionmaker(bind=_engine, autoflush=False, autocommit=False)


def init_db(engine=None) -> None:
    Base.metadata.create_all(bind=engine or _engine)


def get_session() -> Session:
    return SessionLocal()
