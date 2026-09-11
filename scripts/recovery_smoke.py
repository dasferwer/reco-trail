"""После SIGKILL API повторяем подтверждённый запрос и сверяем восстановленную выдачу."""

import json
import subprocess
import time
from pathlib import Path
from uuid import uuid4

from http_client import Client

ROOT = Path(__file__).resolve().parents[1]


def compose(*args):
    subprocess.run(["docker", "compose", *args], cwd=ROOT, check=True, capture_output=True)


def main():
    client = Client("http://localhost:8160")
    email = f"recovery-{uuid4().hex}@example.com"
    client.request(
        "POST", "/auth/register", {"email": email, "password": "RecoTrailDemo123!"}, expected=201
    )
    client.login(email)
    key = uuid4().hex
    body = {"item_id": 1, "kind": "like"}
    event, _ = client.request(
        "POST", "/events", body, headers={"Idempotency-Key": key}, expected=201
    )
    before, _ = client.request("GET", "/recommendations")
    try:
        compose("kill", "--signal", "SIGKILL", "api")
    finally:
        compose("up", "-d", "--no-deps", "--wait", "--wait-timeout", "60", "api")
    for attempt in range(20):
        try:
            client.request("GET", "/health")
            break
        except OSError:
            if attempt == 19:
                raise
            time.sleep(0.5)
    replay, _ = client.request("POST", "/events", body, headers={"Idempotency-Key": key})
    after, _ = client.request("GET", "/recommendations")
    history, _ = client.request("GET", "/events")
    assert replay["event_id"] == event["event_id"] and replay["replayed"]
    assert len(history["items"]) == 1 and before == after
    print(
        json.dumps(
            {
                "ok": True,
                "fault": "SIGKILL api after acknowledged event",
                "events_after_retry": len(history["items"]),
                "stable_recommendations": True,
                "profile_version": after["profile_version"],
                "model_version": after["model_version"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
