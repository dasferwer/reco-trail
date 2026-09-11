"""Проходим путь нового пользователя: предпочтения, рекомендации и обратная связь."""

import json
from datetime import UTC, datetime, timedelta
from uuid import uuid4

from http_client import Client


def main():
    client = Client()
    email = f"smoke-{uuid4().hex}@example.com"
    client.request(
        "POST", "/auth/register", {"email": email, "password": "RecoTrailDemo123!"}, expected=201
    )
    client.login(email)
    popular, _ = client.request("GET", "/recommendations")
    assert popular["strategy"] == "popularity" and len(popular["items"]) == 10
    _, headers = client.request("GET", "/me")
    client.request(
        "PUT", "/me/preferences", {"genres": ["Horror"]}, headers={"If-Match": headers["ETag"]}
    )
    cold, _ = client.request("GET", "/recommendations")
    assert cold["strategy"] == "genre_preferences" and cold["items"] != popular["items"]
    item_id = cold["items"][0]["item_id"]
    key = uuid4().hex
    body = {"item_id": item_id, "kind": "like"}
    event, _ = client.request(
        "POST", "/events", body, headers={"Idempotency-Key": key}, expected=201
    )
    replay, _ = client.request("POST", "/events", body, headers={"Idempotency-Key": key})
    assert replay["event_id"] == event["event_id"] and replay["replayed"]
    late, _ = client.request(
        "POST",
        "/events",
        {
            "item_id": item_id,
            "kind": "dislike",
            "occurred_at": (datetime.now(UTC) - timedelta(days=1)).isoformat(),
        },
        headers={"Idempotency-Key": uuid4().hex},
        expected=201,
    )
    assert not late["applied"] and late["profile_version"] == event["profile_version"]
    warm, _ = client.request("GET", "/recommendations")
    assert warm["strategy"] == "hybrid" and item_id not in [i["item_id"] for i in warm["items"]]
    cached, cache_headers = client.request("GET", "/recommendations")
    assert cached == warm and cache_headers["X-Recommendation-Cache"] == "hit"
    admin = Client()
    admin.login("admin@example.com")
    status, _ = admin.request("PATCH", f"/items/{warm['items'][0]['item_id']}", {"active": False})
    try:
        changed, _ = client.request("GET", "/recommendations")
        assert changed["catalog_version"] > warm["catalog_version"]
        assert status["id"] not in [i["item_id"] for i in changed["items"]]
    finally:
        admin.request("PATCH", f"/items/{status['id']}", {"active": True})
    print(
        json.dumps(
            {
                "ok": True,
                "cold_start": popular["strategy"],
                "preferences": cold["strategy"],
                "after_feedback": warm["strategy"],
                "model_version": warm["model_version"],
                "idempotent_retry": True,
                "late_event_ignored": True,
                "catalog_cache_invalidation": True,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
