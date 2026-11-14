"""
Pattern discovery via clustering.

Clustering runs ONLY over interactions that already have at least one
failure signal attached (deterministic or heuristic) — the goal is to
group failures/edge cases into recurring themes, not to cluster the
entire traffic stream. This keeps clusters meaningful and keeps the
compute cost proportional to what actually needs review.

V1 approach (deliberately simple, per the "don't over-engineer" mandate):
  - TF-IDF over a text representation of each interaction (user_input +
    assistant_output + the names of its attached signal types).
    Including signal types as pseudo-tokens lets interactions with the
    same *kind* of failure cluster together even when their surface text
    differs, without pulling in a vector database or embedding model.
  - HDBSCAN for density-based clustering: no need to pre-specify a
    cluster count, and it naturally produces a "noise" bucket (-1) for
    interactions that don't fit any recurring pattern, which is itself
    useful information (see design_lock cluster-quality rubric: coherence,
    distinctiveness, actionability, representativeness, stability).

This module NEVER reads `metadata["_planted_theme"]` — that field exists
purely for offline evaluation in cluster_quality.py.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

try:
    import hdbscan as _hdbscan_module
    _HAS_HDBSCAN = True
except ImportError:  # pragma: no cover - fallback path
    _HAS_HDBSCAN = False

from sklearn.cluster import DBSCAN

from app.schemas.interaction import InteractionLog
from app.schemas.signal import SignalEvidence

MIN_CLUSTER_SIZE = 3


def _interaction_text(log: InteractionLog, signal_types: list[str]) -> str:
    parts = [log.user_input or "", log.assistant_output or ""]
    # repeat signal type tokens so they carry real TF-IDF weight relative
    # to free text, without hardcoding a fixed embedding scheme
    parts.extend([f"__signal_{t}__" for t in signal_types] * 3)
    return " ".join(parts)


@dataclass
class ClusterAssignment:
    interaction_id: str
    cluster_id: str  # "-1" for noise/unclustered
    probability: float | None = None


@dataclass
class ClusteringResult:
    assignments: list[ClusterAssignment]
    cluster_ids: list[str]  # unique non-noise cluster ids, in order of size desc


def cluster_flagged_interactions(
    logs: list[InteractionLog],
    signals: list[SignalEvidence],
    min_cluster_size: int = MIN_CLUSTER_SIZE,
) -> ClusteringResult:
    """Cluster only the interactions that have >=1 attached signal."""
    signals_by_interaction: dict[str, list[str]] = {}
    for s in signals:
        signals_by_interaction.setdefault(s.interaction_id, []).append(s.signal_type)

    flagged = [log for log in logs if log.interaction_id in signals_by_interaction]

    if len(flagged) < min_cluster_size:
        return ClusteringResult(
            assignments=[ClusterAssignment(log.interaction_id, "-1") for log in flagged],
            cluster_ids=[],
        )

    texts = [_interaction_text(log, signals_by_interaction[log.interaction_id]) for log in flagged]

    vectorizer = TfidfVectorizer(max_features=2000, ngram_range=(1, 2), min_df=1)
    matrix = vectorizer.fit_transform(texts)

    if _HAS_HDBSCAN:
        clusterer = _hdbscan_module.HDBSCAN(min_cluster_size=min_cluster_size, metric="euclidean")
        labels = clusterer.fit_predict(matrix.toarray())
        probabilities = getattr(clusterer, "probabilities_", np.ones(len(labels)))
    else:  # pragma: no cover - fallback if hdbscan unavailable in an environment
        clusterer = DBSCAN(eps=0.8, min_samples=min_cluster_size, metric="cosine")
        labels = clusterer.fit_predict(matrix.toarray())
        probabilities = np.ones(len(labels))

    assignments = [
        ClusterAssignment(
            interaction_id=log.interaction_id,
            cluster_id=str(label),
            probability=float(probabilities[i]) if probabilities is not None else None,
        )
        for i, (log, label) in enumerate(zip(flagged, labels))
    ]

    # order cluster ids by size, descending, for stable dashboard display
    sizes: dict[str, int] = {}
    for a in assignments:
        if a.cluster_id != "-1":
            sizes[a.cluster_id] = sizes.get(a.cluster_id, 0) + 1
    ordered_cluster_ids = sorted(sizes.keys(), key=lambda cid: -sizes[cid])

    return ClusteringResult(assignments=assignments, cluster_ids=ordered_cluster_ids)
