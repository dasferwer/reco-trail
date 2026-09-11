import hashlib
import json
from datetime import UTC, datetime, timedelta

from fastapi import HTTPException
from sqlalchemy import text

from .config import settings
from .db import engine
from .recommendation import load_model


def canonical_hash(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


async def active_model(conn):
    row = (
        (
            await conn.execute(
                text("""
        SELECT s.catalog_version, m.version, m.manifest_sha256
        FROM catalog_settings s JOIN models m ON m.version=s.active_model
    """)
            )
        )
        .mappings()
        .one()
    )
    model = load_model(str(settings.model_dir), row["version"], row["manifest_sha256"])
    return dict(row), model


def validate_genres(genres, model):
    if not set(genres).issubset(model.genres):
        raise HTTPException(422, {"message": "Unknown genre", "allowed": model.genres})
    return sorted(set(genres))


async def record_event(user_id, data, key):
    request_hash = canonical_hash(data.model_dump(mode="json"))
    async with engine.begin() as conn:
        # Одна блокировка упорядочивает события и изменения предпочтений этого пользователя.
        version = (
            await conn.execute(
                text("SELECT profile_version FROM users WHERE id=:user FOR UPDATE"),
                {"user": user_id},
            )
        ).scalar_one()
        previous = (
            (
                await conn.execute(
                    text("""
            SELECT id, request_hash, applied, result_profile_version FROM events
            WHERE user_id=:user AND request_id=:key
        """),
                    {"user": user_id, "key": key},
                )
            )
            .mappings()
            .first()
        )
        if previous:
            if previous["request_hash"] != request_hash:
                raise HTTPException(409, "Idempotency key already used with another payload")
            return {
                "event_id": previous["id"],
                "applied": previous["applied"],
                "profile_version": previous["result_profile_version"],
                "replayed": True,
            }
        if not (
            await conn.execute(text("SELECT active FROM items WHERE id=:id"), {"id": data.item_id})
        ).scalar_one_or_none():
            raise HTTPException(404, "Active item not found")
        now = datetime.now(UTC)
        occurred_at = data.occurred_at or now
        if occurred_at > now + timedelta(seconds=30):
            raise HTTPException(422, "Event timestamp is in the future")
        state = (
            await conn.execute(
                text("""
            SELECT occurred_at FROM user_item_state WHERE user_id=:user AND item_id=:item
        """),
                {"user": user_id, "item": data.item_id},
            )
        ).scalar_one_or_none()
        # Запоздалое событие остаётся в истории, но не отменяет более новое решение пользователя.
        applied = state is None or occurred_at >= state
        result_version = version + int(applied)
        event_id = (
            await conn.execute(
                text("""
            INSERT INTO events(user_id, item_id, kind, occurred_at, request_id, request_hash,
                               applied, result_profile_version)
            VALUES (:user, :item, :kind, :occurred, :key, :hash, :applied, :version) RETURNING id
        """),
                {
                    "user": user_id,
                    "item": data.item_id,
                    "kind": data.kind,
                    "occurred": occurred_at,
                    "key": key,
                    "hash": request_hash,
                    "applied": applied,
                    "version": result_version,
                },
            )
        ).scalar_one()
        if applied:
            await conn.execute(
                text("""
                INSERT INTO user_item_state(user_id, item_id, weight, occurred_at, event_id)
                VALUES (:user, :item, :weight, :occurred, :event)
                ON CONFLICT(user_id, item_id) DO UPDATE SET weight=EXCLUDED.weight,
                    occurred_at=EXCLUDED.occurred_at, event_id=EXCLUDED.event_id
            """),
                {
                    "user": user_id,
                    "item": data.item_id,
                    "weight": {"view": 1.0, "like": 3.0, "dislike": 0.0}[data.kind],
                    "occurred": occurred_at,
                    "event": event_id,
                },
            )
            await conn.execute(
                text("UPDATE users SET profile_version=:version WHERE id=:user"),
                {"version": result_version, "user": user_id},
            )
        return {
            "event_id": event_id,
            "applied": applied,
            "profile_version": result_version,
            "replayed": False,
        }
