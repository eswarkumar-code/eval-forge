"""
Ties clustering output to a CandidateGenerator: builds one ClusterContext
per discovered cluster (capping representative interactions so generation
cost/time stays proportional) and runs the given generator over each.
"""
from __future__ import annotations

from app.candidates.interface import CandidateGenerator, ClusterContext
from app.clustering.cluster import ClusterAssignment
from app.schemas.candidate import EvalCandidate
from app.schemas.interaction import InteractionLog
from app.schemas.signal import SignalEvidence

MAX_REPRESENTATIVES_PER_CLUSTER_CONTEXT = 5


def build_cluster_contexts(
    assignments: list[ClusterAssignment],
    logs: list[InteractionLog],
    signals: list[SignalEvidence],
    cluster_ids: list[str],
) -> list[ClusterContext]:
    logs_by_id = {log.interaction_id: log for log in logs}
    signals_by_interaction: dict[str, list[SignalEvidence]] = {}
    for s in signals:
        signals_by_interaction.setdefault(s.interaction_id, []).append(s)

    by_cluster: dict[str, list[str]] = {}
    for a in assignments:
        if a.cluster_id != "-1":
            by_cluster.setdefault(a.cluster_id, []).append(a.interaction_id)

    contexts = []
    for cid in cluster_ids:
        member_ids = by_cluster.get(cid, [])[:MAX_REPRESENTATIVES_PER_CLUSTER_CONTEXT]
        representative_logs = [logs_by_id[iid] for iid in member_ids if iid in logs_by_id]
        if not representative_logs:
            continue

        signal_type_counts: dict[str, int] = {}
        for iid in member_ids:
            for s in signals_by_interaction.get(iid, []):
                signal_type_counts[s.signal_type] = signal_type_counts.get(s.signal_type, 0) + 1
        dominant_signal_types = sorted(signal_type_counts, key=lambda t: -signal_type_counts[t])

        contexts.append(ClusterContext(
            cluster_id=cid,
            representative_logs=representative_logs,
            signals_by_interaction={iid: signals_by_interaction.get(iid, []) for iid in member_ids},
            dominant_signal_types=dominant_signal_types,
        ))
    return contexts


def generate_candidates_for_clusters(
    generator: CandidateGenerator,
    assignments: list[ClusterAssignment],
    logs: list[InteractionLog],
    signals: list[SignalEvidence],
    cluster_ids: list[str],
) -> list[EvalCandidate]:
    contexts = build_cluster_contexts(assignments, logs, signals, cluster_ids)
    candidates: list[EvalCandidate] = []
    for ctx in contexts:
        candidates.extend(generator.generate(ctx))
    return candidates
