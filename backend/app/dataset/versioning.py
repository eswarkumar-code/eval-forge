"""
Versioned evaluation dataset.

CRITICAL RULE: only candidates that are review_status == "approved" AND
have both reviewer_expected_output and reviewer_failure_category set
(i.e. EvalCandidate.is_canonical_ready() is True) may enter a dataset
version. proposed_* fields alone are never sufficient, no matter how
confident the generator was.

Versions are immutable once created: creating version N+1 never mutates
version N's case list. Each version records its own case_ids, a
timestamp, and category-distribution/provenance summaries for the
dashboard.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from app.schemas.candidate import EvalCandidate


class DatasetVersioningError(ValueError):
    pass


@dataclass(frozen=True)
class DatasetVersion:
    version: int
    created_at: datetime
    case_ids: tuple[str, ...]  # tuple -> immutable
    category_distribution: dict[str, int]
    provenance_distribution: dict[str, int]  # generation_source -> count

    @property
    def approved_case_count(self) -> int:
        return len(self.case_ids)


class EvalDataset:
    """In-memory, append-only version history. A persistence layer (DB)
    can wrap this with the same guarantees; V1 keeps it simple and
    testable without a database dependency."""

    def __init__(self):
        self._versions: list[DatasetVersion] = []
        self._cases_by_id: dict[str, EvalCandidate] = {}  # canonical case store, keyed by candidate_id

    @property
    def versions(self) -> list[DatasetVersion]:
        return list(self._versions)  # defensive copy

    @property
    def latest_version(self) -> Optional[DatasetVersion]:
        return self._versions[-1] if self._versions else None

    def get_case(self, candidate_id: str) -> Optional[EvalCandidate]:
        return self._cases_by_id.get(candidate_id)

    def cases_in_version(self, version_number: int) -> list[EvalCandidate]:
        for v in self._versions:
            if v.version == version_number:
                return [self._cases_by_id[cid] for cid in v.case_ids]
        raise DatasetVersioningError(f"No such version: {version_number}")

    def create_version(self, approved_candidates: list[EvalCandidate]) -> DatasetVersion:
        """Create the next immutable version from the FULL current set of
        approved, canonical-ready candidates (not just newly-approved
        ones) — each version is a complete snapshot, so a reviewer can
        always export "the dataset as it stood at version N" without
        reconstructing deltas."""
        rejected_as_canonical = [c for c in approved_candidates if not c.is_canonical_ready()]
        if rejected_as_canonical:
            raise DatasetVersioningError(
                f"{len(rejected_as_canonical)} candidate(s) passed to create_version are not canonical-ready "
                f"(missing approval or reviewer fields): {[c.candidate_id for c in rejected_as_canonical][:5]}"
            )

        for c in approved_candidates:
            self._cases_by_id[c.candidate_id] = c

        category_dist: dict[str, int] = {}
        provenance_dist: dict[str, int] = {}
        for c in approved_candidates:
            cat = c.reviewer_failure_category or "uncategorized"
            category_dist[cat] = category_dist.get(cat, 0) + 1
            provenance_dist[c.generation_source] = provenance_dist.get(c.generation_source, 0) + 1

        next_version_num = (self._versions[-1].version + 1) if self._versions else 1
        version = DatasetVersion(
            version=next_version_num,
            created_at=datetime.now(timezone.utc),
            case_ids=tuple(c.candidate_id for c in approved_candidates),
            category_distribution=category_dist,
            provenance_distribution=provenance_dist,
        )
        self._versions.append(version)
        return version

    def export_jsonl(self, version_number: int, output_path: Path) -> Path:
        cases = self.cases_in_version(version_number)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w") as f:
            for case in cases:
                f.write(case.model_dump_json() + "\n")
        return output_path
