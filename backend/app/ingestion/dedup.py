"""
Duplicate detection over already-normalized InteractionLog records.

V1 scope: exact-duplicate detection via content_hash (application +
model_name + system_prompt + user_input + assistant_output). This
deliberately does NOT attempt near-duplicate/semantic dedup here — that
overlaps with clustering's job and is a good V2 candidate, not a V1
requirement.

Runs AFTER ingestion, per the locked pipeline order.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from app.schemas.interaction import InteractionLog


@dataclass
class DedupResult:
    unique: list[InteractionLog] = field(default_factory=list)
    duplicates: list[InteractionLog] = field(default_factory=list)
    # maps a duplicate's interaction_id -> the interaction_id of the
    # first-seen record it duplicates
    duplicate_of: dict[str, str] = field(default_factory=dict)


def dedup_interactions(logs: list[InteractionLog]) -> DedupResult:
    result = DedupResult()
    seen: dict[str, str] = {}  # content_hash -> first interaction_id

    for log in logs:
        key = log.content_hash
        if key is None:
            # No hash computed (shouldn't happen via the standard pipeline,
            # but never silently drop a record) -> treat as unique.
            result.unique.append(log)
            continue

        if key in seen:
            result.duplicates.append(log)
            result.duplicate_of[log.interaction_id] = seen[key]
        else:
            seen[key] = log.interaction_id
            result.unique.append(log)

    return result
