# Eval Forge

![Python 3.12](https://img.shields.io/badge/python-3.12-blue) ![FastAPI](https://img.shields.io/badge/backend-FastAPI-009688) ![Tests](https://img.shields.io/badge/tests-132%20passed-brightgreen) ![License MIT](https://img.shields.io/badge/license-MIT-lightgrey)

Automated Eval Dataset Generator from Production Logs — turns raw LLM
interaction logs into a curated, versioned, human-reviewed evaluation
dataset, with regression testing against it.

    Production LLM Logs → Failure Mining → Pattern Discovery →
    Candidate Eval Cases → Human Review → Versioned Eval Dataset →
    Regression Testing

This is a standalone project, fully isolated from any other repository.

## Architecture

```
backend/app/
  schemas/       canonical Pydantic models (InteractionLog, SignalEvidence, EvalCandidate)
  redaction/     deterministic PII detection + stable-placeholder redaction, quarantine triggers
  ingestion/     validate -> redact -> normalize; dedup (exact-hash)
  signals/       deterministic + heuristic failure-signal extraction, provenance-tagged
  clustering/    HDBSCAN over TF-IDF features of flagged interactions; cluster-quality eval
  candidates/    CandidateGenerator interface; deterministic + Claude-backed implementations;
                 candidate-quality rubric checks; cluster -> candidate orchestrator
  review/        human review service — the ONLY path that can mark a candidate canonical
  dataset/       immutable versioned dataset, JSONL export
  regression/    pluggable Judge, pass-rate/category tracking, run-to-run comparison
  db/            SQLAlchemy models + repository (Postgres via DATABASE_URL, SQLite by default)
  api/           FastAPI app wiring the above behind HTTP endpoints
frontend/public/ vanilla HTML/CSS/JS dashboard (no build step) — ingest, clusters, review
                 queue, dataset versions, regression runs, metrics overview
demo_data/       synthetic log generator + generated dataset with planted failure themes
docs/            design_lock.md — the frozen V1 schema/policy/rubric design
scripts/         demo_smoke_test.py — live end-to-end smoke test against a running server
```

## Local startup (no Docker)

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# run the test suite (no network/API key required)
python -m pytest tests/ -v

# run the API (SQLite by default)
uvicorn app.api.main:app --reload
```

Open `frontend/public/index.html` directly in a browser (it talks to
`http://localhost:8000` by default, editable in the sidebar), or serve it
with any static file server.

To ingest the demo dataset:

```bash
curl -X POST http://localhost:8000/ingest \
  -F "file=@../demo_data/synthetic_logs.jsonl;filename=synthetic_logs.jsonl"
```

## Test results

The offline test suite (no network or API key required) currently passes
**132/132 tests**:

```bash
cd backend && python -m pytest tests/ -q
# 132 passed
```

The live Claude acceptance suite in `backend/tests_live/` is opt-in and
excluded from that count (it spends real API tokens).

## Docker (Postgres + backend + frontend)

```bash
docker compose up --build
```

- Frontend: http://localhost:8080
- Backend API: http://localhost:8000
- Postgres: localhost:5432 (user/pass/db: `evalforge`)

To enable the Claude-backed candidate generator in a Docker run, set
`ANTHROPIC_API_KEY` in your shell before `docker compose up`, or in an
untracked `.env` file at the repo root (`docker-compose.yml` reads it via
`${ANTHROPIC_API_KEY:-}`).

**Known limitation:** the development sandbox this project was built in
has no Docker daemon available, so the Dockerfiles/compose file could not
be executed end-to-end in that environment. They were written to the same
conventions as the rest of the stack and reviewed for correctness, but
have not been run. Please validate `docker compose up --build` in your
own environment before relying on it — see
`docs/gate2_docker_checklist.md` for what to check.

## Live Claude acceptance suite (opt-in, costs real API tokens)

The main test suite never calls a real LLM. To exercise the actual
Anthropic-backed candidate generator against the synthetic dataset:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
export RUN_LIVE_CLAUDE_TESTS=1
cd backend && source .venv/bin/activate
pytest tests_live/ -v -s
```

## Important caveat on clustering results

The cluster-quality report in the automated test suite shows purity=1.0
against the synthetic dataset's planted failure themes. **This is a
property of the synthetic data, not a claim about real-production
performance.** The synthetic generator uses a small pool of near-identical
phrasings per planted theme, which makes the TF-IDF + HDBSCAN clustering
approach trivially separate them. Real production logs will have far more
lexical diversity and overlapping failure modes; expect materially lower
purity and higher noise percentage on real data. Treat 1.0 as "the
mechanism works and is wired correctly," not as an accuracy benchmark.
