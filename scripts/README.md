# scripts/

`demo_smoke_test.py` — exercises the live API end-to-end (approve
groundable pending candidates, create a dataset version, run a
regression pass, print dashboard metrics). Run it against a running
server:

    # terminal 1
    cd backend && source .venv/bin/activate && uvicorn app.api.main:app --reload

    # terminal 2 (after POSTing demo_data/synthetic_logs.jsonl to /ingest)
    python3 scripts/demo_smoke_test.py
