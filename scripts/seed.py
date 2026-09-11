"""Первый запуск добавляет каталог и демопрофиль. Повторный не меняет историю пользователя."""

import asyncio
import json
from datetime import UTC, datetime
from uuid import UUID

from sqlalchemy import text

from recotrail.auth import hasher
from recotrail.config import settings
from recotrail.db import engine
from recotrail.recommendation import Model, digest

ADMIN = UUID("16000000-0000-0000-0000-000000000001")
DEMO = UUID("16000000-0000-0000-0000-000000000002")


async def seed():
    directory = settings.model_dir / "movielens-v1"
    model = Model(directory)
    async with engine.begin() as conn:
        await conn.execute(text("SELECT pg_advisory_xact_lock(160001)"))
        for user_id, email, role in [
            (ADMIN, "admin@example.com", "admin"),
            (DEMO, "demo@example.com", "user"),
        ]:
            await conn.execute(
                text("""
                INSERT INTO users(id, email, password_hash, role) VALUES (:id, :email, :hash, :role)
                ON CONFLICT DO NOTHING
            """),
                {
                    "id": user_id,
                    "email": email,
                    "hash": hasher.hash("RecoTrailDemo123!"),
                    "role": role,
                },
            )
        await conn.execute(
            text("""
            INSERT INTO items(id, title, genres) VALUES (:id, :title, CAST(:genres AS jsonb))
            ON CONFLICT DO NOTHING
        """),
            [{**item, "genres": json.dumps(item["genres"])} for item in model.catalog],
        )
        expected = digest(directory / "manifest.json")
        previous = (
            await conn.execute(
                text("SELECT manifest_sha256 FROM models WHERE version=:v"),
                {"v": model.manifest["version"]},
            )
        ).scalar_one_or_none()
        if previous and previous != expected:
            raise ValueError("Existing model versions are immutable")
        await conn.execute(
            text("""
            INSERT INTO models(version, manifest_sha256, evaluation)
            VALUES (:version, :hash, CAST(:evaluation AS jsonb)) ON CONFLICT DO NOTHING
        """),
            {
                "version": model.manifest["version"],
                "hash": expected,
                "evaluation": (directory / "evaluation.json").read_text(),
            },
        )
        await conn.execute(
            text("""
            INSERT INTO catalog_settings(active_model) VALUES (:version) ON CONFLICT DO NOTHING
        """),
            {"version": model.manifest["version"]},
        )
        if not (
            await conn.execute(
                text("SELECT count(*) FROM events WHERE user_id=:user"), {"user": DEMO}
            )
        ).scalar_one():
            for i, event in enumerate(json.loads((directory / "demo-history.json").read_text())):
                occurred = datetime.fromtimestamp(event["timestamp"], UTC)
                event_id = (
                    await conn.execute(
                        text("""
                    INSERT INTO events(user_id, item_id, kind, occurred_at, request_id, request_hash,
                                       applied, result_profile_version)
                    VALUES (:user, :item, 'rating_import', :occurred, :key, 'demo-import', true, :version)
                    RETURNING id
                """),
                        {
                            "user": DEMO,
                            "item": event["item_id"],
                            "occurred": occurred,
                            "key": f"seed-{i}",
                            "version": i + 2,
                        },
                    )
                ).scalar_one()
                await conn.execute(
                    text("""
                    INSERT INTO user_item_state VALUES (:user, :item, :weight, :occurred, :event)
                    ON CONFLICT(user_id, item_id) DO UPDATE SET weight=EXCLUDED.weight,
                        occurred_at=EXCLUDED.occurred_at, event_id=EXCLUDED.event_id
                """),
                    {
                        "user": DEMO,
                        "item": event["item_id"],
                        "weight": event["weight"],
                        "occurred": occurred,
                        "event": event_id,
                    },
                )
            await conn.execute(
                text("UPDATE users SET profile_version=:v WHERE id=:user"),
                {"v": i + 2, "user": DEMO},
            )
    print(json.dumps({"model": model.manifest["version"], "catalog_items": len(model.catalog)}))


if __name__ == "__main__":
    asyncio.run(seed())
