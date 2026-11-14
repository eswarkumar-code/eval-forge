# Live Claude Acceptance Suite

Not part of the main automated test suite. Exercises the real Anthropic
API via `ClaudeCandidateGenerator` against synthetic/redacted records only.

## Run

    export ANTHROPIC_API_KEY=sk-ant-...
    export RUN_LIVE_CLAUDE_TESTS=1
    cd backend && source .venv/bin/activate
    pytest tests_live/ -v -s

Costs real API tokens. Uses only demo_data/synthetic_logs.jsonl, capped to
a small number of clusters/interactions per test to keep cost minimal.
