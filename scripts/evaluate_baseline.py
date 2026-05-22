#!/usr/bin/env python3
"""Evaluate the Day 5 baseline retrieval quality and latency."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mm_search_lab.config import load_yaml  # noqa: E402
from mm_search_lab.metrics import round_metrics, summarize_retrieval  # noqa: E402
from mm_search_lab.search import SearchService  # noqa: E402


def parse_ks(value: str) -> List[int]:
    ks = sorted({int(part.strip()) for part in value.split(",") if part.strip()})
    if not ks or any(k <= 0 for k in ks):
        raise ValueError("--ks must contain positive integers")
    return ks


def evaluate_text_to_image(service: SearchService, top_k: int) -> List[dict]:
    records = []
    for item in service.catalog:
        query = item.title
        start = time.perf_counter()
        results = service.search_text(query, top_k=top_k)
        latency_ms = (time.perf_counter() - start) * 1000.0
        records.append(
            {
                "task": "text_to_image",
                "query_id": item.id,
                "query": query,
                "relevant_ids": [item.id],
                "latency_ms": latency_ms,
                "results": results,
            }
        )
    return records


def evaluate_image_to_image(service: SearchService, top_k: int) -> List[dict]:
    records = []
    for item in service.catalog:
        image_path = str(item.image_path)
        start = time.perf_counter()
        results = service.search_image(image_path, top_k=top_k)
        latency_ms = (time.perf_counter() - start) * 1000.0
        records.append(
            {
                "task": "image_to_image",
                "query_id": item.id,
                "image_path": image_path,
                "relevant_ids": [item.id],
                "latency_ms": latency_ms,
                "results": results,
            }
        )
    return records


def write_jsonl(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--ks", default="1,5")
    parser.add_argument("--tasks", default="text_to_image,image_to_image")
    args = parser.parse_args()

    config = load_yaml(args.config)
    paths = config["paths"]
    ks = parse_ks(args.ks)
    top_k = max(ks)
    tasks = {task.strip() for task in args.tasks.split(",") if task.strip()}

    report_dir = ROOT / paths["report_dir"]
    report_dir.mkdir(parents=True, exist_ok=True)

    service = SearchService(args.config, ROOT)
    warmup_start = time.perf_counter()
    service.search_text(service.catalog[0].title, top_k=1)
    warmup_ms = (time.perf_counter() - warmup_start) * 1000.0

    task_records = {}
    if "text_to_image" in tasks:
        task_records["text_to_image"] = evaluate_text_to_image(service, top_k)
    if "image_to_image" in tasks:
        task_records["image_to_image"] = evaluate_image_to_image(service, top_k)
    if not task_records:
        raise ValueError("--tasks did not include any supported task")

    metrics = {
        "config": str(Path(args.config)),
        "ks": ks,
        "model": service.model_config,
        "index": service.health(),
        "warmup_ms": round(warmup_ms, 4),
        "tasks": {},
    }

    all_records = []
    for task_name, records in task_records.items():
        metrics["tasks"][task_name] = round_metrics(summarize_retrieval(records, ks))
        all_records.extend(records)

    metrics["overall"] = round_metrics(summarize_retrieval(all_records, ks))

    metrics_path = report_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    write_jsonl(report_dir / "per_query_results.jsonl", all_records)

    print("evaluated tasks: {}".format(", ".join(sorted(task_records))))
    print("warmup_ms: {:.4f}".format(warmup_ms))
    print("num_queries: {}".format(len(all_records)))
    print("wrote metrics: {}".format(metrics_path))
    print(json.dumps(metrics["overall"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
