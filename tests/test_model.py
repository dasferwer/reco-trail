import json
import shutil

import numpy as np
import pytest
from conftest import ROOT
from threadpoolctl import threadpool_limits

from recotrail.recommendation import Model, digest, load_model
from scripts.train import evaluate, ranking_metrics, read_data


def test_metrics_match_hand_calculated_example():
    metric = ranking_metrics([10, 20, 30], {10, 30, 40}, k=3)
    assert metric["recall"] == pytest.approx(2 / 3)
    assert metric["ndcg"] == pytest.approx(1.5 / (1 + 1 / np.log2(3) + 0.5))
    with pytest.raises(ValueError):
        ranking_metrics([], set())


def test_global_temporal_split_excludes_future_catalog_and_events(model):
    _, train, validation, test, _ = read_data(ROOT / "data")
    assert max(r[0] for r in train) < min(r[0] for r in validation)
    assert max(r[0] for r in validation) < min(r[0] for r in test)
    assert set(model.item_index) == {r[2] for r in train + validation}
    assert model.manifest["fit_rows"] == len(train) + len(validation)
    assert model.manifest["fit_cutoff_exclusive"] <= min(r[0] for r in test)


def test_committed_test_metrics_recompute_from_original_ratings(model):
    _, train, validation, test, _ = read_data(ROOT / "data")
    report = json.loads((ROOT / "models/movielens-v1/evaluation.json").read_text())
    with threadpool_limits(limits=2):
        for algorithm, expected in report["test"].items():
            actual = evaluate(model, train + validation, test, algorithm)
            for key in ["recall_at_10", "ndcg_at_10", "catalog_coverage", "cold_ndcg_at_10"]:
                assert actual[key] == pytest.approx(expected[key], abs=1e-12)
            assert actual["warm_users"] == expected["warm_users"]
    selected = max(
        report["validation_algorithms"],
        key=lambda k: report["validation_algorithms"][k]["ndcg_at_10"],
    )
    assert selected == model.manifest["default_algorithm"]


def test_checksum_rejects_modified_numeric_model(tmp_path):
    shutil.copytree(ROOT / "models/movielens-v1", tmp_path / "copy")
    with (tmp_path / "copy/weights.npz").open("ab") as stream:
        stream.write(b"tampered")
    with pytest.raises(ValueError, match="checksum"):
        Model(tmp_path / "copy")
    with pytest.raises(ValueError, match="Invalid model version"):
        load_model(str(tmp_path), "../copy", "x")


def test_manifest_cannot_omit_checksums(tmp_path):
    shutil.copytree(ROOT / "models/movielens-v1", tmp_path / "copy")
    path = tmp_path / "copy/manifest.json"
    manifest = json.loads(path.read_text())
    manifest["files_sha256"] = {}
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="every serving artifact"):
        Model(tmp_path / "copy")
    with pytest.raises(ValueError, match="differs"):
        load_model(str(tmp_path), "copy", digest(ROOT / "models/movielens-v1/manifest.json"))


def test_seen_and_inactive_items_never_enter_ranking(model):
    catalog = [dict(i) for i in model.catalog]
    seen = {catalog[0]["id"]: 3, catalog[1]["id"]: 0}
    for item in catalog[2:5]:
        item["active"] = False
    result = model.rank(seen, catalog, limit=10000)
    assert not {i["item_id"] for i in result["items"]} & {i["id"] for i in catalog[:5]}


def test_cold_start_and_zero_positive_history_use_popularity(model):
    for algorithm in ["content", "collaborative", "hybrid", "popularity"]:
        assert model.rank({}, model.catalog, algorithm=algorithm)["strategy"] == "popularity"
    assert model.rank({1: 0}, model.catalog)["strategy"] == "popularity"
    assert model.rank({}, model.catalog, preferences=["Horror"])["strategy"] == "genre_preferences"
    assert model.rank({}, [])["items"] == []


def test_group_report_counts_errors_and_keeps_original_aggregate(model):
    _, train, validation, test, _ = read_data(ROOT / "data")
    with threadpool_limits(limits=2):
        report = evaluate(model, train + validation, test, "hybrid")
    for group in ["warm", "cold"]:
        rows = report["groups"][group]
        assert rows["users"] == report[group + "_users"]
        assert rows["hits_at_10"] + rows["missed_relevant_items"] == rows["relevant_items"]
        assert rows["zero_hit_users"] == sum(r["hits"] == 0 for r in rows["per_user"])
    assert report["groups"]["warm"]["ndcg_at_10"] == report["ndcg_at_10"]
    assert report["groups"]["cold"]["ndcg_at_10"] == report["cold_ndcg_at_10"]


def test_stress_catalog_eligibility_uses_real_ranking(model):
    from scripts.evaluate_groups import stress_catalog

    items, history = stress_catalog(model)
    with threadpool_limits(limits=2):
        result = model.rank(history, items, limit=10000)
    ids = {row["item_id"] for row in result["items"]}
    assert len(items) == 10000
    assert len(ids) == 9800
    assert ids == {row["id"] for row in items if row["active"] and row["id"] not in history}
    assert stress_catalog(model) == (items, history)
