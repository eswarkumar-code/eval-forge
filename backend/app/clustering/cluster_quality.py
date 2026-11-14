"""
Offline cluster-quality evaluation against the synthetic dataset's planted
failure themes (design_lock.md section 4). This is NOT part of the
production pipeline — it's a validation tool that requires ground truth
(_planted_theme) that only exists in the demo dataset.

Metrics computed, exactly as specified in the design lock:
  - cluster purity against planted themes
  - noise/unclustered percentage
  - dominant-theme percentage per cluster

We are explicitly NOT optimizing for perfect scores. The goal is to show
whether clustering surfaces recognizable recurring patterns at all.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from app.clustering.cluster import ClusterAssignment
from app.schemas.interaction import InteractionLog


@dataclass
class ClusterThemeBreakdown:
    cluster_id: str
    size: int
    dominant_theme: str | None
    dominant_theme_pct: float
    theme_counts: dict[str, int] = field(default_factory=dict)


@dataclass
class ClusterQualityReport:
    total_flagged: int
    noise_count: int
    noise_pct: float
    num_clusters: int
    overall_purity: float  # weighted avg of dominant_theme_pct across clusters, by cluster size
    clusters: list[ClusterThemeBreakdown]


def evaluate_cluster_quality(
    assignments: list[ClusterAssignment],
    logs_by_id: dict[str, InteractionLog],
) -> ClusterQualityReport:
    total = len(assignments)
    noise = [a for a in assignments if a.cluster_id == "-1"]
    non_noise = [a for a in assignments if a.cluster_id != "-1"]

    by_cluster: dict[str, list[ClusterAssignment]] = {}
    for a in non_noise:
        by_cluster.setdefault(a.cluster_id, []).append(a)

    breakdowns: list[ClusterThemeBreakdown] = []
    weighted_purity_sum = 0.0

    for cid, members in by_cluster.items():
        themes = [
            (logs_by_id[m.interaction_id].metadata or {}).get("_planted_theme") or "unlabeled"
            for m in members
        ]
        counts = Counter(themes)
        dominant_theme, dominant_count = counts.most_common(1)[0]
        dominant_pct = dominant_count / len(members)
        breakdowns.append(ClusterThemeBreakdown(
            cluster_id=cid,
            size=len(members),
            dominant_theme=dominant_theme,
            dominant_theme_pct=round(dominant_pct, 3),
            theme_counts=dict(counts),
        ))
        weighted_purity_sum += dominant_pct * len(members)

    breakdowns.sort(key=lambda b: -b.size)

    overall_purity = round(weighted_purity_sum / len(non_noise), 3) if non_noise else 0.0
    noise_pct = round(len(noise) / total, 3) if total else 0.0

    return ClusterQualityReport(
        total_flagged=total,
        noise_count=len(noise),
        noise_pct=noise_pct,
        num_clusters=len(by_cluster),
        overall_purity=overall_purity,
        clusters=breakdowns,
    )
