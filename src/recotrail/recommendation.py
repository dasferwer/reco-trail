import hashlib
import json
from functools import lru_cache
from pathlib import Path

import numpy as np


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def scale(values):
    values = np.asarray(values, dtype=np.float64)
    span = float(np.max(values) - np.min(values)) if len(values) else 0
    return (values - np.min(values)) / span if span > 1e-12 else np.zeros_like(values)


class Model:
    def __init__(self, directory):
        directory = Path(directory)
        self.manifest = json.loads((directory / "manifest.json").read_text())
        if self.manifest["format"] != 1:
            raise ValueError("Unsupported model format")
        if set(self.manifest["files_sha256"]) != {"catalog.json", "weights.npz"}:
            raise ValueError("Model manifest must cover every serving artifact")
        for name, expected in self.manifest["files_sha256"].items():
            if name not in {"catalog.json", "weights.npz"} or digest(directory / name) != expected:
                raise ValueError("Model checksum mismatch")
        self.catalog = json.loads((directory / "catalog.json").read_text())
        self.genres = self.manifest["genres"]
        self.genre_index = {name: i for i, name in enumerate(self.genres)}
        self.item_index = {item["id"]: i for i, item in enumerate(self.catalog)}
        with np.load(directory / "weights.npz", allow_pickle=False) as arrays:
            self.factors = arrays["factors"]
            self.popularity = arrays["popularity"]
        self.alpha = float(self.manifest["alpha"])
        if (
            self.factors.shape != (len(self.catalog), self.manifest["factors"])
            or self.popularity.shape != (len(self.catalog),)
            or not np.isfinite(self.factors).all()
            or not np.isfinite(self.popularity).all()
            or np.any(self.popularity < 0)
            or not 0 <= self.alpha <= 1
        ):
            raise ValueError("Invalid model dimensions or numeric values")

    def scores(self, history, items, preferences=(), algorithm="hybrid"):
        count = len(items)
        if not count:
            return np.array([]), "empty_catalog"
        content = np.zeros((count, len(self.genres)), np.float64)
        factors = np.zeros((count, self.factors.shape[1]), np.float64)
        popularity = np.zeros(count, np.float64)
        for i, item in enumerate(items):
            for genre in item["genres"]:
                if genre in self.genre_index:
                    content[i, self.genre_index[genre]] = 1
            index = self.item_index.get(item["id"])
            if index is not None:
                factors[i] = self.factors[index]
                popularity[i] = self.popularity[index]
        norms = np.linalg.norm(content, axis=1, keepdims=True)
        content /= np.maximum(norms, 1)
        profile = np.zeros(len(self.genres), np.float64)
        latent = np.zeros(self.factors.shape[1], np.float64)
        positives = 0
        item_metadata = {item["id"]: item for item in self.catalog}
        item_metadata.update({item["id"]: item for item in items})
        for item_id, weight in history.items():
            if weight <= 0:
                continue
            positives += 1
            index = self.item_index.get(item_id)
            if index is not None:
                latent += weight * self.factors[index]
            for genre in item_metadata.get(item_id, {}).get("genres", []):
                if genre in self.genre_index:
                    profile[self.genre_index[genre]] += weight
        for genre in preferences:
            if genre in self.genre_index:
                profile[self.genre_index[genre]] += 3
        profile /= max(float(np.linalg.norm(profile)), 1)
        pop = scale(np.log1p(popularity))
        by_content = scale(content @ profile)
        collaborative = scale(factors @ latent)
        if algorithm == "popularity" or (positives == 0 and not preferences):
            return pop, "popularity"
        if positives == 0 or np.linalg.norm(latent) < 1e-12:
            return (
                0.95 * by_content + 0.05 * pop,
                "genre_preferences" if preferences else "content_fallback",
            )
        if algorithm == "content":
            return 0.95 * by_content + 0.05 * pop, "content"
        if algorithm == "collaborative":
            return 0.95 * collaborative + 0.05 * pop, "collaborative"
        return 0.95 * (
            self.alpha * collaborative + (1 - self.alpha) * by_content
        ) + 0.05 * pop, "hybrid"

    def rank(self, history, items, preferences=(), algorithm="hybrid", limit=10):
        scores, strategy = self.scores(history, items, preferences, algorithm)
        # Просмотренные и отвергнутые фильмы исключаются после расчёта профиля интересов.
        available = [
            i
            for i, item in enumerate(items)
            if item["id"] not in history and item.get("active", True)
        ]
        available.sort(key=lambda i: (-float(scores[i]), items[i]["id"]))
        results = []
        positive_genres = set(preferences)
        metadata = {item["id"]: item for item in self.catalog}
        metadata.update({item["id"]: item for item in items})
        for item_id, weight in history.items():
            if weight > 0:
                positive_genres.update(metadata.get(item_id, {}).get("genres", []))
        for i in available[:limit]:
            item = items[i]
            matched = sorted(set(item["genres"]) & positive_genres)
            results.append(
                {
                    "item_id": item["id"],
                    "title": item["title"],
                    "genres": item["genres"],
                    "score": round(float(scores[i]), 8),
                    "matched_genres": matched[:3],
                    "reason": "Похожие жанры в ваших предпочтениях и истории"
                    if matched and strategy != "popularity"
                    else "Похожие оценки в обучающей выборке"
                    if strategy in {"collaborative", "hybrid"}
                    else "Популярно в обучающей выборке",
                }
            )
        return {"strategy": strategy, "items": results}


@lru_cache(maxsize=2)
def load_model(root, version, expected_hash):
    import re

    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", version):
        raise ValueError("Invalid model version")
    directory = Path(root) / version
    if digest(directory / "manifest.json") != expected_hash:
        raise ValueError("Registered model manifest differs")
    return Model(directory)
