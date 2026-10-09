"""Пересчёт замороженного temporal holdout и локальный ресурсный замер без обучения."""

import argparse
import json
import os
import platform
import resource
import time
import tracemalloc
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_info, threadpool_limits

from recotrail.recommendation import Model, digest
from scripts.train import ROOT, evaluate, read_data


def stress_catalog(model, size=10000, seed=216):
    if size < len(model.catalog):
        raise ValueError("Stress catalog cannot truncate frozen catalog")
    rng = np.random.default_rng(seed)
    items = [dict(row, active=True) for row in model.catalog]
    next_id = max(model.item_index) + 1
    for i in range(size - len(items)):
        items.append(
            {
                "id": next_id + i,
                "title": f"Synthetic stress item {i}",
                "genres": [str(rng.choice(model.genres))],
                "active": True,
            }
        )
    # Непересекающиеся seen/inactive позволяют проверить точное число кандидатов.
    history = {row["id"]: (2 if i % 2 else 0) for i, row in enumerate(items[:100])}
    for row in items[100:200]:
        row["active"] = False
    return items, history


def benchmark(model, repeats=100):
    if repeats < 20:
        raise ValueError("At least 20 measured repetitions required")
    items, history = stress_catalog(model)
    algorithm = model.manifest["default_algorithm"]
    excluded = set(history) | {r["id"] for r in items if not r["active"]}
    eligible = {r["id"] for r in items} - excluded
    full = model.rank(history, items, algorithm=algorithm, limit=len(items))
    assert {r["item_id"] for r in full["items"]} == eligible
    expected = model.rank(history, items, algorithm=algorithm)
    for _ in range(5):
        assert model.rank(history, items, algorithm=algorithm) == expected
    timings = []
    cpu_started = time.process_time()
    for _ in range(repeats):
        started = time.perf_counter()
        result = model.rank(history, items, algorithm=algorithm)
        timings.append((time.perf_counter() - started) * 1000)
        assert result == expected
        assert not {r["item_id"] for r in result["items"]} & excluded
    cpu_seconds = time.process_time() - cpu_started
    # Отдельный проход: tracemalloc меняет время выполнения, поэтому не включён в p95.
    tracemalloc.start()
    model.rank(history, items, algorithm=algorithm)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rss_bytes = rss if platform.system() == "Darwin" else rss * 1024
    return {
        "seed": 216,
        "catalog_size": len(items),
        "real_model_items": len(model.catalog),
        "synthetic_untrained_items": len(items) - len(model.catalog),
        "seen_items": len(history),
        "inactive_items": 100,
        "eligible_items": len(eligible),
        "top_k": 10,
        "algorithm": algorithm,
        "strategy": expected["strategy"],
        "warmup": 5,
        "repetitions": repeats,
        "concurrency": 1,
        "p50_ms": float(np.percentile(timings, 50)),
        "p95_ms": float(np.percentile(timings, 95)),
        "max_ms": max(timings),
        "cpu_seconds": cpu_seconds,
        "ranking_tracemalloc_peak_bytes": peak,
        "process_lifetime_peak_rss_bytes": rss_bytes,
        "scope": "In-process Model.rank; excludes HTTP, DB, model load and cache; synthetic extension has no trained factors",
        "invariants": "Full ranking equals all 9800 eligible IDs; seen/inactive absent; all top10 repeats identical",
    }


def main(output, repeats):
    directory = ROOT / "models/movielens-v1"
    model = Model(directory)
    _, train, validation, test, source = read_data(ROOT / "data")
    past = train + validation
    assert set(model.item_index) == {row[2] for row in past}
    assert model.manifest["fit_cutoff_exclusive"] <= min(row[0] for row in test)
    with threadpool_limits(limits=2):
        report = {
            "created_at": datetime.now(UTC).isoformat(),
            "version": model.manifest["version"],
            "manifest_sha256": digest(directory / "manifest.json"),
            "source": source,
            "protocol": "Frozen original global temporal test; no tuning or refitting on test labels. Warm = positive past rating, cold = no positive past rating; cold can have rejected items.",
            "split_rows": {"past": len(past), "test": len(test)},
            "test_start": min(row[0] for row in test),
            "algorithms": {
                name: evaluate(model, past, test, name)
                for name in ["popularity", "content", "collaborative", "hybrid"]
            },
            "environment": {
                "platform": platform.platform(),
                "machine": platform.machine(),
                "python": platform.python_version(),
                "numpy": np.__version__,
                "logical_cpus": os.cpu_count(),
                "threadpools": threadpool_info(),
            },
            "benchmark": benchmark(model, repeats),
        }
    Path(output).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report["benchmark"], indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="docs/p2-ranking.json")
    parser.add_argument("--repeats", type=int, default=100)
    args = parser.parse_args()
    main(args.output, args.repeats)
