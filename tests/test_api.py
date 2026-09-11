import asyncio
import json
import shutil
from datetime import UTC, datetime, timedelta

import pytest
from conftest import ADMIN, OTHER, ROOT, headers
from sqlalchemy import text

from recotrail.config import settings
from recotrail.db import engine
from recotrail.recommendation import digest


async def send(client, item=1, kind="like", key="event-1", occurred=None):
    body = {"item_id": item, "kind": kind}
    if occurred:
        body["occurred_at"] = occurred.isoformat()
    return await client.post("/events", json=body, headers={"Idempotency-Key": key})


async def test_auth_and_administrator_boundary(client):
    assert (await client.get("/me", headers={"Authorization": "Bearer broken"})).status_code == 401
    assert (await client.get("/models")).status_code == 403
    assert (await client.post("/items", json={"title": "Test", "genres": []})).status_code == 403
    assert (await client.get("/models", headers=headers(ADMIN))).status_code == 200
    registered = await client.post(
        "/auth/register", json={"email": "new@example.com", "password": "LongEnough123!"}
    )
    assert registered.status_code == 201
    token = (
        await client.post(
            "/auth/login", json={"email": "new@example.com", "password": "LongEnough123!"}
        )
    ).json()["access_token"]
    assert (await client.get("/me", headers={"Authorization": "Bearer " + token})).json()[
        "email"
    ] == "new@example.com"


async def test_concurrent_identical_events_apply_once(client):
    responses = await asyncio.gather(*(send(client) for _ in range(12)))
    assert sorted(r.status_code for r in responses) == [200] * 11 + [201]
    assert len({r.json()["event_id"] for r in responses}) == 1
    assert (await client.get("/me")).json()["profile_version"] == 2
    assert len((await client.get("/events")).json()["items"]) == 1


async def test_conflicting_retry_does_not_change_state(client):
    assert (await send(client)).status_code == 201
    assert (await send(client, kind="dislike")).status_code == 409
    async with engine.connect() as conn:
        assert (await conn.execute(text("SELECT weight FROM user_item_state"))).scalar_one() == 3


async def test_distinct_concurrent_events_have_serial_versions(client):
    responses = await asyncio.gather(*(send(client, key=f"event-{i}") for i in range(10)))
    assert sorted(r.json()["profile_version"] for r in responses) == list(range(2, 12))
    assert (await client.get("/me")).json()["profile_version"] == 11


async def test_late_event_is_retained_without_replacing_dislike(client):
    now = datetime.now(UTC)
    await send(client, kind="dislike", occurred=now)
    late = await send(client, kind="like", key="late", occurred=now - timedelta(days=1))
    assert late.json()["applied"] is False
    assert late.json()["profile_version"] == 2
    async with engine.connect() as conn:
        assert (await conn.execute(text("SELECT weight FROM user_item_state"))).scalar_one() == 0
    assert len((await client.get("/events")).json()["items"]) == 2


async def test_equal_timestamp_uses_commit_order(client):
    now = datetime.now(UTC)
    await send(client, occurred=now)
    await send(client, kind="dislike", key="second", occurred=now)
    async with engine.connect() as conn:
        assert (await conn.execute(text("SELECT weight FROM user_item_state"))).scalar_one() == 0


async def test_invalid_events_leave_no_history(client):
    assert (await send(client, item=999999)).status_code == 404
    assert (await send(client, occurred=datetime.now(UTC) + timedelta(days=1))).status_code == 422
    assert (await client.post("/events", json={"item_id": 1, "kind": "like"})).status_code == 422
    assert (
        await client.post(
            "/events",
            headers={"Idempotency-Key": "a"},
            json={"item_id": 1, "kind": "like", "occurred_at": "2020-01-01T12:00:00"},
        )
    ).status_code == 422
    assert (await client.get("/events")).json()["items"] == []


async def test_history_and_cache_are_private(client):
    await send(client)
    first = (await client.get("/recommendations")).json()
    other = (await client.get("/recommendations", headers=headers(OTHER))).json()
    assert first["profile_version"] == 2 and other["profile_version"] == 1
    assert (await client.get("/events", headers=headers(OTHER))).json()["items"] == []
    assert (await client.get("/me", headers=headers(OTHER))).json()["preferences"] == []


async def test_preferences_require_fresh_etag(client):
    assert (await client.put("/me/preferences", json={"genres": ["Horror"]})).status_code == 428
    etag = (await client.get("/me")).headers["ETag"]
    results = await asyncio.gather(
        *(
            client.put("/me/preferences", headers={"If-Match": etag}, json={"genres": [genre]})
            for genre in ["Horror", "Comedy"]
        )
    )
    assert sorted(r.status_code for r in results) == [200, 412]
    assert (
        await client.put(
            "/me/preferences", headers={"If-Match": '"2"'}, json={"genres": ["invented"]}
        )
    ).status_code == 422


async def test_cache_invalidates_after_preferences_and_events(client):
    first = await client.get("/recommendations")
    assert first.headers["X-Recommendation-Cache"] == "miss"
    assert first.json()["strategy"] == "popularity"
    assert (await client.get("/recommendations")).headers["X-Recommendation-Cache"] == "hit"
    await client.put("/me/preferences", headers={"If-Match": '"1"'}, json={"genres": ["Horror"]})
    second = await client.get("/recommendations")
    assert second.headers["X-Recommendation-Cache"] == "miss"
    assert second.json()["strategy"] == "genre_preferences"
    assert second.json()["items"] != first.json()["items"]
    item_id = second.json()["items"][0]["item_id"]
    await send(client, item=item_id)
    third = await client.get("/recommendations")
    assert third.headers["X-Recommendation-Cache"] == "miss"
    assert item_id not in [i["item_id"] for i in third.json()["items"]]


async def test_catalog_updates_invalidate_cache_and_hide_inactive_items(client):
    first = (await client.get("/recommendations")).json()
    item_id = first["items"][0]["item_id"]
    await client.patch(f"/items/{item_id}", headers=headers(ADMIN), json={"active": False})
    result = await client.get("/recommendations")
    assert result.headers["X-Recommendation-Cache"] == "miss"
    assert result.json()["catalog_version"] == first["catalog_version"] + 1
    assert item_id not in [i["item_id"] for i in result.json()["items"]]
    assert (await send(client, item=item_id)).status_code == 404


async def test_new_item_can_be_recommended_without_latent_factors(client):
    response = await client.post(
        "/items", headers=headers(ADMIN), json={"title": "A new western", "genres": ["Western"]}
    )
    assert response.status_code == 201
    item_id = response.json()["id"]
    async with engine.begin() as conn:
        await conn.execute(text("UPDATE items SET active=false WHERE id!=:id"), {"id": item_id})
        await conn.execute(text("UPDATE catalog_settings SET catalog_version=catalog_version+1"))
    await client.put("/me/preferences", headers={"If-Match": '"1"'}, json={"genres": ["Western"]})
    result = (await client.get("/recommendations")).json()
    assert result["items"][0]["item_id"] == item_id
    assert result["items"][0]["matched_genres"] == ["Western"]
    assert result["items"][0]["score"] > 0


async def test_atomic_rollback_when_profile_write_fails(client):
    async with engine.begin() as conn:
        await conn.execute(
            text("""
            CREATE FUNCTION fail_profile_write() RETURNS trigger LANGUAGE plpgsql AS $$
            BEGIN RAISE EXCEPTION 'simulated storage failure'; END $$
        """)
        )
        await conn.execute(
            text(
                "CREATE TRIGGER fail_profile BEFORE UPDATE ON users FOR EACH ROW EXECUTE FUNCTION fail_profile_write()"
            )
        )
    try:
        with pytest.raises(Exception, match="simulated storage failure"):
            await send(client)
        async with engine.connect() as conn:
            assert (await conn.execute(text("SELECT count(*) FROM events"))).scalar_one() == 0
            assert (
                await conn.execute(text("SELECT count(*) FROM user_item_state"))
            ).scalar_one() == 0
    finally:
        async with engine.begin() as conn:
            await conn.execute(text("DROP TRIGGER fail_profile ON users"))
            await conn.execute(text("DROP FUNCTION fail_profile_write()"))
    assert (await send(client)).status_code == 201


async def test_model_activation_switches_cache_key(client, tmp_path, monkeypatch):
    shutil.copytree(ROOT / "models/movielens-v1", tmp_path / "movielens-v1")
    alternate = tmp_path / "movielens-alternate"
    shutil.copytree(ROOT / "models/movielens-v1", alternate)
    manifest = json.loads((alternate / "manifest.json").read_text())
    manifest.update(version="movielens-alternate", default_algorithm="popularity")
    (alternate / "manifest.json").write_text(json.dumps(manifest))
    monkeypatch.setattr(settings, "model_dir", tmp_path)
    async with engine.begin() as conn:
        await conn.execute(
            text("INSERT INTO models VALUES ('movielens-alternate', :hash, '{}', now())"),
            {"hash": digest(alternate / "manifest.json")},
        )
    await client.get("/recommendations")
    assert (
        await client.post("/models/missing/activate", headers=headers(ADMIN))
    ).status_code == 404
    assert (
        await client.post("/models/movielens-alternate/activate", headers=headers(ADMIN))
    ).status_code == 200
    result = await client.get("/recommendations")
    assert result.headers["X-Recommendation-Cache"] == "miss"
    assert result.json()["model_version"] == "movielens-alternate"
    assert result.json()["algorithm"] == "popularity"


async def test_history_pagination_has_no_gaps(client):
    for i in range(5):
        await send(client, key=f"event-{i}")
    after, ids = 0, []
    while True:
        result = (await client.get(f"/events?after={after}&limit=2")).json()
        if not result["items"]:
            break
        ids.extend(row["id"] for row in result["items"])
        after = result["next_after"]
    assert len(ids) == len(set(ids)) == 5


async def test_seed_is_repeatable_and_preserves_user_changes(client):
    from scripts.seed import DEMO, seed

    await seed()
    async with engine.begin() as conn:
        await conn.execute(
            text("UPDATE users SET preferences='[\"Horror\"]', profile_version=500 WHERE id=:id"),
            {"id": DEMO},
        )
        count = (await conn.execute(text("SELECT count(*) FROM events"))).scalar_one()
    await seed()
    async with engine.connect() as conn:
        assert (await conn.execute(text("SELECT count(*) FROM events"))).scalar_one() == count
        assert (
            await conn.execute(text("SELECT profile_version FROM users WHERE id=:id"), {"id": DEMO})
        ).scalar_one() == 500


async def test_recommendation_snapshot_matches_its_profile_version(client):
    first = (await client.get("/recommendations")).json()["items"][0]["item_id"]
    requests = [client.get("/recommendations") for _ in range(6)] + [send(client, item=first)]
    results = await asyncio.gather(*requests)
    for response in results[:-1]:
        data = response.json()
        ids = [i["item_id"] for i in data["items"]]
        assert (first in ids) == (data["profile_version"] == 1)
    assert (await client.get("/recommendations")).json()["profile_version"] == 2
