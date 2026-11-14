# Gate 2 — Docker Acceptance Checklist (run on your Mac)

Run this in a terminal window that is **not** in the Ledger or OPEN LLM
directories — `cd` into a fresh clone/copy of `eval-forge` only.

Docker was not available in the sandbox this project was built in, so
none of this has been executed yet. The Dockerfiles and compose file
follow the same conventions as the rest of the stack and were reviewed
for correctness, but this is the first real run. Go through the checklist
in order; if any step fails, stop and report the exact output rather than
working around it.

## 0. Prerequisites

```bash
docker --version
docker compose version
```

Both should return version strings. If `docker compose` (v2, no hyphen)
isn't available, use `docker-compose` instead in the commands below.

## 1. Clean state

```bash
cd /path/to/eval-forge
docker compose down -v --remove-orphans 2>/dev/null  # harmless if nothing exists yet
docker system df   # note current disk usage, for comparison after teardown later
```

## 2. Build and start

```bash
docker compose up --build
```

**Watch for, in order:**
- [ ] `postgres` container logs show `database system is ready to accept connections`
- [ ] `backend` container logs show `Application startup complete` (FastAPI/uvicorn) — this only happens *after* Postgres's healthcheck passes, since `backend` has `depends_on: postgres: condition: service_healthy`
- [ ] `frontend` container starts (nginx) with no errors
- [ ] No container exits/restarts in a loop (`docker compose ps` should show all three as `Up`/`running`, not `Restarting`)

If `backend` fails to build, the most likely cause is the `hdbscan` wheel needing a C compiler — the Dockerfile installs `build-essential` for exactly this reason; if it still fails, capture the full build log.

Leave this running in this terminal, open a second terminal for the rest.

## 3. Health and connectivity

```bash
curl -s http://localhost:8000/health
```
- [ ] Returns `{"status":"ok","time":"..."}`

```bash
docker compose exec backend python3 -c "from app.db.session import make_engine; e = make_engine(); print(e.connect())"
```
- [ ] Prints a connection object with no traceback — confirms the backend container can actually reach Postgres (not just that Postgres is up)

## 4. Frontend loads

Open `http://localhost:8080` in a browser.
- [ ] Dashboard loads (dark UI, "evalforge" wordmark in the sidebar, Overview tab active)
- [ ] Overview shows all-zero metrics initially (no error state) — confirms the frontend is correctly calling the backend's `/dashboard/metrics` at `http://localhost:8000` from your browser (cross-origin, should work given `allow_origins=["*"]`)

If the Overview panel shows "Could not reach API", check the API endpoint field in the sidebar — it defaults to `http://localhost:8000`, which should be correct for this compose setup since the backend port is published to the host.

## 5. Ingestion + failure mining + clustering

From the dashboard's **Ingest** tab, upload `demo_data/synthetic_logs.jsonl`, or via curl:

```bash
curl -s -X POST http://localhost:8000/ingest \
  -F "file=@demo_data/synthetic_logs.jsonl;filename=synthetic_logs.jsonl"
```
- [ ] Returns JSON with `accepted_count` ~118, `quarantined_count` ~9, `clusters_found` ~10, `new_candidates_generated` ~46 (exact numbers may vary slightly by environment but should be in this range — the underlying test suite pins exact numbers if you need to compare precisely)

Go to the **Clusters** tab in the dashboard:
- [ ] Multiple clusters appear, each with example input/output pairs and a proposed category

Go to **Overview**:
- [ ] Metrics now show non-zero logs/signals/clusters/candidates

## 6. Candidate review

Go to the **Review** tab (Pending filter):
- [ ] Candidates appear, each showing input, observed output, and a dashed amber "Proposed by system — not yet reviewed, not canonical" block
- [ ] Approve one candidate that has a proposed expected output
- [ ] After approving, switch to the Approved filter — the approved candidate now shows a green "✓ Human-approved — canonical ground truth" block
- [ ] Try approving a candidate with no proposed expected output — this should fail (400 error), confirming the system won't let you approve without grounding

## 7. Dataset versioning + export

Go to the **Dataset** tab:
- [ ] Click "Create new version from approved cases" — should succeed if you approved at least one candidate in step 6
- [ ] Version `1` appears in the version list with the correct approved-case count
- [ ] Click "export JSONL" — downloads/opens a JSONL file; open it and confirm each line has `review_status: "approved"` and non-null `reviewer_expected_output`/`reviewer_failure_category`

## 8. Regression execution

Go to the **Regression** tab:
- [ ] Click "Run regression against latest dataset version" — should succeed and show a pass rate
- [ ] Run history shows the run

## 9. Persistence survives restart

```bash
docker compose restart backend
sleep 5
curl -s http://localhost:8000/dashboard/metrics
```
- [ ] Metrics still show the same non-zero counts as before the restart — this proves state is in Postgres, not lost from an in-memory store, on an in-place container restart

Now a harder test — full container **recreation** (not just restart), keeping the volume:

```bash
docker compose down          # NOTE: no -v flag — this preserves the postgres volume
docker compose up --build
sleep 5
curl -s http://localhost:8000/dashboard/metrics
```
- [ ] Metrics are still intact after full container recreation, confirming durability is coming from the named Postgres volume (`evalforge_pgdata`), not container filesystem state

## 10. Fresh teardown / rebuild

```bash
docker compose down -v --remove-orphans   # -v this time: also removes the postgres volume
docker compose up --build
sleep 5
curl -s http://localhost:8000/dashboard/metrics
```
- [ ] This time metrics should be back to **zero** — confirms `-v` genuinely wiped persisted state and the rebuild starts clean, i.e. you're not accidentally testing against stale cached data

```bash
docker compose down -v --remove-orphans
```
- [ ] Final cleanup — all containers, networks, and the named volume removed

## Report back

For each numbered section above, note pass/fail and paste any error output
verbatim (don't summarize/paraphrase tracebacks) — that lets me fix root
cause rather than guessing.
