"""Сравниваем рекомендации на будущих событиях, не перемешивая историю случайным образом."""

import argparse
import csv
import json
import time
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from threadpoolctl import threadpool_limits

from recotrail.recommendation import Model, digest

ROOT = Path(__file__).resolve().parents[1]


def read_data(directory):
    source = json.loads((directory / "source.json").read_text())
    for name, expected in source["files_sha256"].items():
        if digest(directory / name) != expected:
            raise ValueError("Dataset checksum mismatch: " + name)
    with (directory / "movies.csv").open() as stream:
        movies = {
            int(row["movieId"]): {
                "id": int(row["movieId"]),
                "title": row["title"],
                "genres": [
                    genre for genre in row["genres"].split("|") if genre != "(no genres listed)"
                ],
            }
            for row in csv.DictReader(stream)
        }
    with (directory / "ratings.csv").open() as stream:
        ratings = [
            (int(r["timestamp"]), int(r["userId"]), int(r["movieId"]), float(r["rating"]))
            for r in csv.DictReader(stream)
        ]
    ratings.sort()
    cutoff1 = ratings[int(len(ratings) * 0.8)][0]
    cutoff2 = ratings[int(len(ratings) * 0.9)][0]
    train = [r for r in ratings if r[0] < cutoff1]
    validation = [r for r in ratings if cutoff1 <= r[0] < cutoff2]
    test = [r for r in ratings if r[0] >= cutoff2]
    assert max(r[0] for r in train) < min(r[0] for r in validation)
    assert max(r[0] for r in validation) < min(r[0] for r in test)
    return movies, train, validation, test, source


def histories(rows):
    result = defaultdict(dict)
    for _, user, item, rating in sorted(rows):
        result[user][item] = max(rating - 3, 0)
    return result


def fit(rows, movies, directory, version, alpha):
    ids = sorted({r[2] for r in rows})
    users = sorted({r[1] for r in rows})
    item_index = {item: i for i, item in enumerate(ids)}
    user_index = {user: i for i, user in enumerate(users)}
    history = histories(rows)
    coords = [
        (user_index[user], item_index[item], weight)
        for user, values in history.items()
        for item, weight in values.items()
        if weight > 0
    ]
    matrix = csr_matrix(
        ([x[2] for x in coords], ([x[0] for x in coords], [x[1] for x in coords])),
        shape=(len(users), len(ids)),
        dtype=np.float64,
    )
    # Матрица содержит только события до cutoff. Будущие оценки не влияют на факторы и популярность.
    svd = TruncatedSVD(n_components=32, n_iter=7, random_state=42).fit(matrix)
    factors = svd.components_.T
    popularity = np.asarray((matrix > 0).sum(axis=0)).ravel().astype(np.float64)
    catalog = [movies[item] for item in ids]
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "catalog.json").write_text(json.dumps(catalog, ensure_ascii=False) + "\n")
    np.savez_compressed(directory / "weights.npz", factors=factors, popularity=popularity)
    manifest = {
        "format": 1,
        "version": version,
        "alpha": alpha,
        "genres": sorted({g for item in catalog for g in item["genres"]}),
        "fit_rows": len(rows),
        "fit_cutoff_exclusive": max(r[0] for r in rows) + 1,
        "fit_items": len(ids),
        "factors": 32,
        "seed": 42,
        "files_sha256": {
            name: digest(directory / name) for name in ["catalog.json", "weights.npz"]
        },
    }
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return Model(directory)


def ranking_metrics(predicted, relevant, k=10):
    relevant = set(relevant)
    if not relevant:
        raise ValueError("Evaluation needs at least one relevant item")
    hits = [1 if item in relevant else 0 for item in predicted[:k]]
    dcg = sum(hit / np.log2(i + 2) for i, hit in enumerate(hits))
    ideal = sum(1 / np.log2(i + 2) for i in range(min(k, len(relevant))))
    return {"recall": sum(hits) / len(relevant), "ndcg": dcg / ideal, "hits": sum(hits)}


def evaluate(model, past, future, algorithm):
    history = histories(past)
    truth = defaultdict(set)
    known = {item["id"] for item in model.catalog}
    positive_total = 0
    unavailable = 0
    already_seen = 0
    for _, user, item, rating in future:
        if rating < 4:
            continue
        positive_total += 1
        if item not in known:
            unavailable += 1
            continue
        if item in history.get(user, {}):
            already_seen += 1
            continue
        truth[user].add(item)
    values = []
    recommended = set()
    cold = []
    groups = {name: [] for name in ("warm", "cold")}
    for user, relevant in sorted(truth.items()):
        profile = history.get(user, {})
        result = model.rank(profile, model.catalog, algorithm=algorithm)
        ids = [row["item_id"] for row in result["items"]]
        metric = ranking_metrics(ids, relevant)
        if not any(weight > 0 for weight in profile.values()):
            cold.append(metric)
        else:
            values.append(metric)
        group = "warm" if any(weight > 0 for weight in profile.values()) else "cold"
        groups[group].append(
            {
                "user_id": user,
                "relevant_items": len(relevant),
                "recommended_items": len(ids),
                **metric,
            }
        )
        assert not set(ids) & set(profile), "Seen items leaked into evaluation"
        recommended.update(ids)

    def average(rows, key):
        return float(np.mean([row[key] for row in rows])) if rows else 0.0

    return {
        "groups": {
            name: {
                "users": len(rows),
                "relevant_items": sum(r["relevant_items"] for r in rows),
                "hits_at_10": sum(r["hits"] for r in rows),
                "missed_relevant_items": sum(r["relevant_items"] - r["hits"] for r in rows),
                "zero_hit_users": sum(r["hits"] == 0 for r in rows),
                "recall_at_10": average(rows, "recall"),
                "ndcg_at_10": average(rows, "ndcg"),
                "per_user": rows,
            }
            for name, rows in groups.items()
        },
        "warm_users": len(values),
        "cold_users": len(cold),
        "recall_at_10": average(values, "recall"),
        "ndcg_at_10": average(values, "ndcg"),
        "cold_recall_at_10": average(cold, "recall"),
        "cold_ndcg_at_10": average(cold, "ndcg"),
        "catalog_coverage": len(recommended) / len(known),
        "future_positive_events": positive_total,
        "future_items_unavailable": unavailable,
        "future_already_seen": already_seen,
        "candidate_catalog_size": len(known),
        "candidate_protocol": "Full known catalog; every previously rated item excluded; no sampled negatives",
    }


def main(version):
    import re
    import tempfile

    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", version):
        raise ValueError("Invalid version")
    directory = ROOT / "models" / version
    if directory.exists():
        raise ValueError("Version already exists")
    started = time.monotonic()
    movies, train, validation, test, source = read_data(ROOT / "data")
    with tempfile.TemporaryDirectory() as temporary:
        model = fit(train, movies, Path(temporary) / "validation", "validation", 0.5)
        trials = []
        for alpha in [0, 0.25, 0.5, 0.75, 1]:
            model.alpha = alpha
            metric = evaluate(model, train, validation, "hybrid")
            trials.append({"alpha": alpha, **metric})
        selected = max(trials, key=lambda row: (row["ndcg_at_10"], -row["alpha"]))["alpha"]
        model.alpha = selected
        baseline_validation = {
            name: evaluate(model, train, validation, name)
            for name in ["popularity", "content", "collaborative", "hybrid"]
        }
        selected_algorithm = max(
            baseline_validation, key=lambda name: baseline_validation[name]["ndcg_at_10"]
        )
    final = fit(train + validation, movies, directory, version, selected)
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["default_algorithm"] = selected_algorithm
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    models = {
        name: evaluate(final, train + validation, test, name)
        for name in ["popularity", "content", "collaborative", "hybrid"]
    }
    report = {
        "version": version,
        "created_at": datetime.now(UTC).isoformat(),
        "dataset": source,
        "split_rows": {"train": len(train), "validation": len(validation), "test": len(test)},
        "cutoffs": {
            "validation_start": min(r[0] for r in validation),
            "test_start": min(r[0] for r in test),
        },
        "selected_alpha": selected,
        "selected_algorithm": selected_algorithm,
        "validation_algorithms": baseline_validation,
        "selection": "Highest validation NDCG@10 on warm users; final factors refitted on train+validation before test",
        "validation_trials": trials,
        "test": models,
        "elapsed_seconds": round(time.monotonic() - started, 2),
    }
    (directory / "evaluation.json").write_text(json.dumps(report, indent=2) + "\n")
    history = histories(train + validation)
    demo_user = max(history, key=lambda user: sum(v > 0 for v in history[user].values()))
    latest = [r for r in train + validation if r[1] == demo_user][-40:]
    (directory / "demo-history.json").write_text(
        json.dumps(
            [{"item_id": r[2], "weight": max(r[3] - 3, 0), "timestamp": r[0]} for r in latest],
            indent=2,
        )
        + "\n"
    )
    print(
        json.dumps(
            {
                "version": version,
                "alpha": selected,
                "split_rows": report["split_rows"],
                "elapsed_seconds": report["elapsed_seconds"],
                "test": models,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", default="movielens-v1")
    args = parser.parse_args()
    with threadpool_limits(limits=2):
        main(args.version)
