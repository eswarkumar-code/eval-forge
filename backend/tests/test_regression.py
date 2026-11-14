from datetime import datetime, timezone

from app.regression.runner import SimilarityJudge, compare_runs, run_regression
from app.review.service import approve_candidate
from app.schemas.candidate import EvalCandidate


def _approved_candidate(cid, expected, category="general"):
    c = EvalCandidate(
        candidate_id=cid,
        source_interaction_ids=["i1"],
        input_text="some input",
        observed_output="bad output",
        proposed_expected_output=expected,
        proposed_failure_category=category,
        generation_source="user_correction",
        created_at=datetime.now(timezone.utc),
    )
    return approve_candidate(c)


def test_run_regression_only_evaluates_canonical_ready_cases():
    approved = _approved_candidate("c1", "the correct answer")
    pending = EvalCandidate(
        candidate_id="c2", source_interaction_ids=["i2"], input_text="x",
        proposed_expected_output="y", proposed_failure_category="z",
        generation_source="deterministic", created_at=datetime.now(timezone.utc),
    )  # never approved
    run = run_regression([approved, pending], lambda c: "the correct answer", dataset_version=1)
    assert len(run.results) == 1
    assert run.results[0].case_id == "c1"


def test_perfect_match_passes():
    approved = _approved_candidate("c1", "the correct answer")
    run = run_regression([approved], lambda c: "the correct answer", dataset_version=1)
    assert run.pass_rate == 1.0
    assert run.results[0].passed


def test_completely_wrong_output_fails():
    approved = _approved_candidate("c1", "the correct answer")
    run = run_regression([approved], lambda c: "completely unrelated nonsense response text", dataset_version=1)
    assert run.pass_rate == 0.0
    assert not run.results[0].passed


def test_pass_rate_and_category_breakdown():
    cases = [
        _approved_candidate("c1", "answer one", category="cat_a"),
        _approved_candidate("c2", "answer two", category="cat_a"),
        _approved_candidate("c3", "answer three", category="cat_b"),
    ]
    def sut(case):
        return "answer one" if case.candidate_id == "c1" else "wrong wrong wrong"
    run = run_regression(cases, sut, dataset_version=1)
    assert run.pass_rate == round(1 / 3, 3)
    assert run.failures_by_category == {"cat_a": 1, "cat_b": 1}


def test_compare_runs_with_no_previous_run():
    approved = _approved_candidate("c1", "answer")
    run = run_regression([approved], lambda c: "answer", dataset_version=1)
    comparison = compare_runs(run, previous=None)
    assert comparison.previous_run_id is None
    assert comparison.new_cases == ["c1"]
    assert comparison.regressions == []


def test_compare_runs_detects_regression():
    approved = _approved_candidate("c1", "answer")
    previous_run = run_regression([approved], lambda c: "answer", dataset_version=1)  # passes
    current_run = run_regression([approved], lambda c: "totally different wrong text", dataset_version=1)  # now fails
    comparison = compare_runs(current_run, previous_run)
    assert comparison.regressions == ["c1"]
    assert comparison.improvements == []
    assert comparison.pass_rate_delta == -1.0


def test_compare_runs_detects_improvement():
    approved = _approved_candidate("c1", "answer")
    previous_run = run_regression([approved], lambda c: "totally different wrong text", dataset_version=1)  # fails
    current_run = run_regression([approved], lambda c: "answer", dataset_version=1)  # now passes
    comparison = compare_runs(current_run, previous_run)
    assert comparison.improvements == ["c1"]
    assert comparison.regressions == []
    assert comparison.pass_rate_delta == 1.0


def test_custom_judge_can_be_swapped_in():
    approved = _approved_candidate("c1", "answer")

    class AlwaysPassJudge(SimilarityJudge):
        def judge(self, case, actual_output):
            return True, 1.0

    run = run_regression([approved], lambda c: "irrelevant", dataset_version=1, judge=AlwaysPassJudge())
    assert run.pass_rate == 1.0
