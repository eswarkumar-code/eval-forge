# Project #13 — Design Lock (frozen before implementation)

This is the authoritative design reference cited throughout the codebase
as `docs/design_lock.md`. It was locked before implementation began and
implementation follows it exactly; deviations require updating this file
first.

## 1. Normalized Production Log Schema

Every raw interaction must be converted into one canonical internal
record before anything else runs. See `backend/app/schemas/interaction.py`
for the implementation (`InteractionLog`).

Key rules:
- A retry is not the same thing as a correction. Retry = another model
  attempt for substantially the same request. Correction = a human/user
  explicitly provides or edits what should have happened.
- A multi-turn conversation is never collapsed into one giant string.
  `conversation_id` + `interaction_id` + `parent_interaction_id` preserve
  sequence while still allowing each interaction to be analyzed
  individually.
- Every downstream object must maintain provenance back to the original
  interaction ID(s).

## 2. Sensitive-Data Policy

Sensitive data is detected and redacted during ingestion, before
persistence, clustering, candidate generation, model analysis, or export.
The unredacted original is never stored by this application.

- Detection is deterministic (regex-based) for V1: email, phone, IP,
  account/card-style numbers, SSN-style patterns, API keys/tokens/
  credentials, URLs with embedded secrets. Comprehensive PII detection is
  not claimed.
- Redaction uses stable typed placeholders (`<EMAIL_1>`, `<PHONE_1>`, ...).
  Repeated occurrences of the same raw value within one interaction map
  to the same placeholder, preserving structure useful for clustering.
- Order of operations, strictly: raw input → validate → sensitive-data
  scan → redact → persist normalized record → dedup → failure mining →
  clustering → candidate generation. Never cluster or send model-assisted
  analysis over raw unredacted input.
- Quarantine (rather than auto-redact-and-process) for records containing
  credentials, tokens, private keys, or an unusually dense concentration
  of sensitive identifiers. Quarantined records store only enough
  metadata to explain why they were quarantined.

See `backend/app/redaction/redactor.py`.

## 3. Failure Signal Provenance

Every detected signal has an explicit source, enforced in the actual
schema (`backend/app/schemas/signal.py`), not just documentation:
- `deterministic` — HTTP/API error, malformed JSON, thumbs down, retry
  occurred, user edited answer.
- `heuristic` — latency outlier, large response divergence after retry,
  possible refusal mismatch.
- `model_generated` — suggested failure category, suggested ideal
  response, semantic critique.
- `human_approved` — reviewer-confirmed failure category, corrected
  expected output, canonical eval label.

Only the final human-approved category becomes ground truth for the
evaluation dataset.

## 4. Cluster Quality Rubric

Clustering is exploratory; clusters are not claimed to be objectively
correct. A cluster is useful when it satisfies most of: coherence,
distinctiveness, actionability, representative examples (not a single
anomaly), and stability under reasonable parameter changes.

The synthetic demo dataset plants known failure themes (incorrect date
reasoning, malformed structured JSON, refusal when answer should be
allowed, unsupported factual claims, missing required fields, wrong
classification, excessive latency/retry, incomplete multi-part answers)
specifically so clustering can be evaluated against something real:
purity against planted themes, noise/unclustered percentage, and
dominant-theme percentage per cluster. Perfect scores are not the goal —
demonstrating that clustering surfaces recognizable recurring patterns
is. See `backend/app/clustering/cluster_quality.py` and the caveat in the
top-level README about synthetic-data purity not generalizing to
production performance.

## 5. Candidate Eval Quality Rubric

A candidate eval case is good only if it is reproducible (original
input/context sufficient to reproduce the failure), specific (not a vague
complaint), groundable (enough evidence to know what better behavior
looks like), non-duplicative, representative (recurring pattern or
important high-severity edge case), reviewable (a human can approve/
reject without reconstructing production history), and properly sourced
(expected output/category clearly attributed to user correction,
deterministic rule, model suggestion, or human review). See
`backend/app/candidates/quality.py`.

## 6. Candidate Eval Schema

See `backend/app/schemas/candidate.py` (`EvalCandidate`).

Critical rule: `proposed_expected_output` / `proposed_failure_category`
never automatically become canonical ground truth. Only
`reviewer_expected_output` / `reviewer_failure_category`, set via an
actual human review action with `review_status == "approved"`, may enter
the versioned dataset. Enforced structurally via
`EvalCandidate.is_canonical_ready()` and by `backend/app/review/service.py`
being the only code path allowed to set those fields.

## 7. Build Order (locked)

Schema + redaction → ingestion → dedup/quarantine → deterministic/
heuristic failure signals → planted synthetic dataset → clustering →
cluster-quality evaluation → candidate generation → candidate-quality
evaluation → human review → immutable dataset versions → regression
runner → dashboard last.

## Final design principle

The system does not claim that heuristics, clustering, or model-generated
labels are ground truth. They are prioritization tools. Human-reviewed
cases are the only canonical evaluation data.
