"""
Regression runner.

Evaluates a "system under test" (any callable: input_text -> actual
output) against one version of the canonical eval dataset. V1 uses a
deterministic text-similarity judge by default (no API dependency, fully
offline-testable) — see SimilarityJudge. The Judge interface is
pluggable so a stricter/LLM-based judge can be swapped in later without
changing the runner.

Tracks, per design_lock/spec requirements:
  - pass rate
  - failures by category
  - regressions (passed in previous run, fails now)
  - improvements (failed in previous run, passes now)
  - comparison against the previous run
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from difflib import SequenceMatcher
from typing import Callable, Optional
from uuid import uuid4

from app.schemas.candidate import EvalCandidate

SystemUnderTest = Callable[[EvalCandidate], str]


class Judge(ABC):
    @abstractmethod
    def judge(self, case: EvalCandidate, actual_output: str) -> tuple[bool, float]:
        """Return (passed, score) where score is in [0, 1]."""
        ...


class SimilarityJudge(Judge):
    """Deterministic, offline judge: passes if actual output is
    sufficiently textually similar to reviewer_expected_output. Coarse by
    design — good enough to demonstrate the regression-tracking mechanics
    without requiring a model-based judge. A stricter/semantic judge is a
    reasonable V2 upgrade, not a V1 requirement."""

    def __init__(self, threshold: float = 0.6):
        self.threshold = threshold

    def judge(self, case: EvalCandidate, actual_output: str) -> tuple[bool, float]:
        expected = case.reviewer_expected_output or ""
        ratio = SequenceMatcher(None, expected.lower().strip(), (actual_output or "").lower().strip()).ratio()
        return ratio >= self.threshold, round(ratio, 3)


@dataclass
class CaseResult:
    case_id: str
    category: str
    passed: bool
    score: float
    actual_output: str


@dataclass
class RegressionRun:
    run_id: str
    dataset_version: int
    created_at: datetime
    results: list[CaseResult] = field(default_factory=list)

    @property
    def pass_rate(self) -> float:
        if not self.results:
            return 0.0
        return round(sum(1 for r in self.results if r.passed) / len(self.results), 3)

    @property
    def failures_by_category(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for r in self.results:
            if not r.passed:
                counts[r.category] = counts.get(r.category, 0) + 1
        return counts

    @property
    def results_by_case_id(self) -> dict[str, CaseResult]:
        return {r.case_id: r for r in self.results}


@dataclass
class RegressionComparison:
    current_run_id: str
    previous_run_id: Optional[str]
    pass_rate_delta: float
    regressions: list[str]  # case_ids that passed before, fail now
    improvements: list[str]  # case_ids that failed before, pass now
    new_cases: list[str]  # case_ids present now but not in previous run


def run_regression(
    cases: list[EvalCandidate],
    system_under_test: SystemUnderTest,
    dataset_version: int,
    judge: Optional[Judge] = None,
) -> RegressionRun:
    judge = judge or SimilarityJudge()
    run = RegressionRun(run_id=str(uuid4()), dataset_version=dataset_version, created_at=datetime.now(timezone.utc))

    for case in cases:
        if not case.is_canonical_ready():
            # regression must only ever run against canonical, reviewer-approved
            # cases — never against a proposed_* field that a human never signed off on.
            continue
        actual_output = system_under_test(case)
        passed, score = judge.judge(case, actual_output)
        run.results.append(CaseResult(
            case_id=case.candidate_id,
            category=case.reviewer_failure_category or "uncategorized",
            passed=passed,
            score=score,
            actual_output=actual_output,
        ))

    return run


def compare_runs(current: RegressionRun, previous: Optional[RegressionRun]) -> RegressionComparison:
    if previous is None:
        return RegressionComparison(
            current_run_id=current.run_id,
            previous_run_id=None,
            pass_rate_delta=0.0,
            regressions=[],
            improvements=[],
            new_cases=[r.case_id for r in current.results],
        )

    prev_by_id = previous.results_by_case_id
    curr_by_id = current.results_by_case_id

    regressions = [cid for cid, r in curr_by_id.items() if cid in prev_by_id and prev_by_id[cid].passed and not r.passed]
    improvements = [cid for cid, r in curr_by_id.items() if cid in prev_by_id and not prev_by_id[cid].passed and r.passed]
    new_cases = [cid for cid in curr_by_id if cid not in prev_by_id]

    return RegressionComparison(
        current_run_id=current.run_id,
        previous_run_id=previous.run_id,
        pass_rate_delta=round(current.pass_rate - previous.pass_rate, 3),
        regressions=regressions,
        improvements=improvements,
        new_cases=new_cases,
    )
