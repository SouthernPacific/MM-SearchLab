"""Offline retrieval metrics for MM-SearchLab."""

from __future__ import annotations

import math
from statistics import mean
from typing import Dict, Iterable, List, Sequence, Set


def find_first_relevant_rank(results: Sequence[dict], relevant_ids: Iterable[str]) -> int:
    relevant: Set[str] = set(relevant_ids)
    for row in results:
        if row.get("id") in relevant:
            return int(row.get("rank", 0))
    return 0


def recall_at_k(results: Sequence[dict], relevant_ids: Iterable[str], k: int) -> float:
    if k <= 0:
        raise ValueError("k must be positive")
    relevant: Set[str] = set(relevant_ids)
    if not relevant:
        return 0.0
    returned = {row.get("id") for row in results[:k]}
    return 1.0 if returned & relevant else 0.0


def reciprocal_rank(results: Sequence[dict], relevant_ids: Iterable[str]) -> float:
    rank = find_first_relevant_rank(results, relevant_ids)
    return 1.0 / rank if rank > 0 else 0.0


def ndcg_at_k(results: Sequence[dict], relevant_ids: Iterable[str], k: int) -> float:
    if k <= 0:
        raise ValueError("k must be positive")
    relevant: Set[str] = set(relevant_ids)
    if not relevant:
        return 0.0
    dcg = 0.0
    for idx, row in enumerate(results[:k], start=1):
        if row.get("id") in relevant:
            dcg += 1.0 / math.log2(idx + 1)
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(idx + 1) for idx in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


def latency_summary(latencies_ms: Sequence[float]) -> Dict[str, float]:
    if not latencies_ms:
        return {"avg_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0, "max_ms": 0.0}
    values = sorted(float(value) for value in latencies_ms)
    return {
        "avg_ms": mean(values),
        "p50_ms": percentile(values, 50),
        "p95_ms": percentile(values, 95),
        "max_ms": values[-1],
    }


def percentile(sorted_values: Sequence[float], pct: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    rank = (len(sorted_values) - 1) * pct / 100.0
    low = int(math.floor(rank))
    high = int(math.ceil(rank))
    if low == high:
        return float(sorted_values[low])
    weight = rank - low
    return float(sorted_values[low] * (1.0 - weight) + sorted_values[high] * weight)


def summarize_retrieval(records: Sequence[dict], ks: Sequence[int]) -> Dict[str, float]:
    if not records:
        raise ValueError("records must not be empty")
    summary: Dict[str, float] = {"num_queries": float(len(records))}
    for k in ks:
        summary["recall@{}".format(k)] = mean(
            recall_at_k(record["results"], record["relevant_ids"], k) for record in records
        )
        summary["ndcg@{}".format(k)] = mean(
            ndcg_at_k(record["results"], record["relevant_ids"], k) for record in records
        )
    summary["mrr"] = mean(reciprocal_rank(record["results"], record["relevant_ids"]) for record in records)
    summary.update(latency_summary([record["latency_ms"] for record in records]))
    return summary


def round_metrics(metrics: Dict[str, float], digits: int = 4) -> Dict[str, float]:
    rounded: Dict[str, float] = {}
    for key, value in metrics.items():
        if key == "num_queries":
            rounded[key] = int(value)
        else:
            rounded[key] = round(float(value), digits)
    return rounded
