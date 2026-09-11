import asyncio
import json
from collections import OrderedDict
from typing import Annotated, Literal

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Response
from pydantic import AwareDatetime, BaseModel, Field, field_validator
from sqlalchemy import text

from .auth import User, admin_user, current_user
from .auth import router as auth_router
from .db import engine
from .observability import instrument
from .service import active_model, record_event, validate_genres

app = FastAPI(
    title="RecoTrail", version="0.1.0", description="Recommendations with versioned user feedback"
)
app.include_router(auth_router)
instrument(app)
cache = OrderedDict()
inference_slots = asyncio.Semaphore(2)


class EventInput(BaseModel):
    item_id: int = Field(gt=0)
    kind: Literal["view", "like", "dislike"]
    occurred_at: AwareDatetime | None = None


class Preferences(BaseModel):
    genres: list[str] = Field(max_length=19)


class NewItem(Preferences):
    title: str = Field(min_length=1, max_length=300)

    @field_validator("title")
    @classmethod
    def valid_title(cls, value):
        if not value.strip() or any(ord(char) < 32 for char in value):
            raise ValueError("Title must contain visible text without control characters")
        return value.strip()


class ItemStatus(BaseModel):
    active: bool


def match_version(header, version):
    if header is None:
        raise HTTPException(428, "If-Match is required; read the current profile ETag first")
    if header != f'"{version}"':
        raise HTTPException(412, "Profile changed; read it again before updating")


@app.get("/health", tags=["Operations"])
async def health():
    async with engine.connect() as conn:
        info, _ = await active_model(conn)
    return {"status": "ok", "model_version": info["version"]}


@app.get("/me", tags=["Profile"])
async def profile(response: Response, user: User = Depends(current_user)):
    async with engine.connect() as conn:
        row = (
            (
                await conn.execute(
                    text("""
            SELECT id, email, preferences, profile_version FROM users WHERE id=:id
        """),
                    {"id": user.id},
                )
            )
            .mappings()
            .one()
        )
    response.headers["ETag"] = f'"{row["profile_version"]}"'
    return dict(row)


@app.put("/me/preferences", tags=["Profile"])
async def preferences(
    data: Preferences,
    response: Response,
    if_match: str | None = Header(default=None),
    user: User = Depends(current_user),
):
    async with engine.begin() as conn:
        row = (
            (
                await conn.execute(
                    text("""
            SELECT profile_version, preferences FROM users WHERE id=:id FOR UPDATE
        """),
                    {"id": user.id},
                )
            )
            .mappings()
            .one()
        )
        match_version(if_match, row["profile_version"])
        _, model = await active_model(conn)
        genres = validate_genres(data.genres, model)
        version = row["profile_version"] + int(genres != row["preferences"])
        await conn.execute(
            text("""
            UPDATE users SET preferences=CAST(:genres AS jsonb), profile_version=:version WHERE id=:id
        """),
            {"genres": json.dumps(genres), "version": version, "id": user.id},
        )
    response.headers["ETag"] = f'"{version}"'
    return {"genres": genres, "profile_version": version}


@app.post("/events", status_code=201, tags=["Feedback"])
async def event(
    data: EventInput,
    response: Response,
    idempotency_key: Annotated[
        str, Header(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_.:-]+$")
    ],
    user: User = Depends(current_user),
):
    result = await record_event(user.id, data, idempotency_key)
    if result["replayed"]:
        response.status_code = 200
    return result


@app.get("/events", tags=["Feedback"])
async def events(
    after: int = Query(default=0, ge=0),
    limit: int = Query(default=50, ge=1, le=100),
    user: User = Depends(current_user),
):
    async with engine.connect() as conn:
        rows = (
            (
                await conn.execute(
                    text("""
            SELECT id, item_id, kind, occurred_at, created_at, applied FROM events
            WHERE user_id=:user AND id>:after ORDER BY id LIMIT :limit
        """),
                    {"user": user.id, "after": after, "limit": limit},
                )
            )
            .mappings()
            .all()
        )
    return {"items": [dict(row) for row in rows], "next_after": rows[-1]["id"] if rows else after}


@app.get("/genres", tags=["Catalog"])
async def genres(user: User = Depends(current_user)):
    async with engine.connect() as conn:
        _, model = await active_model(conn)
    return {"genres": model.genres}


@app.get("/items", tags=["Catalog"])
async def items(
    after: int = Query(default=0, ge=0),
    limit: int = Query(default=50, ge=1, le=100),
    genre: str | None = Query(default=None, max_length=30),
    user: User = Depends(current_user),
):
    async with engine.connect() as conn:
        rows = (
            (
                await conn.execute(
                    text("""
            SELECT id, title, genres FROM items WHERE active AND id>:after
            AND (CAST(:genre AS text) IS NULL OR genres ? CAST(:genre AS text)) ORDER BY id LIMIT :limit
        """),
                    {"after": after, "limit": limit, "genre": genre},
                )
            )
            .mappings()
            .all()
        )
    return {"items": [dict(row) for row in rows], "next_after": rows[-1]["id"] if rows else after}


@app.post("/items", status_code=201, tags=["Catalog administration"])
async def create_item(data: NewItem, user: User = Depends(admin_user)):
    async with engine.begin() as conn:
        await conn.execute(text("SELECT singleton FROM catalog_settings FOR UPDATE"))
        if (await conn.execute(text("SELECT count(*) FROM items"))).scalar_one() >= 10000:
            raise HTTPException(409, "Demo catalog limit reached (10000)")
        _, model = await active_model(conn)
        genres = validate_genres(data.genres, model)
        row = (
            (
                await conn.execute(
                    text("""
            INSERT INTO items(title, genres) VALUES (:title, CAST(:genres AS jsonb)) RETURNING *
        """),
                    {"title": data.title, "genres": json.dumps(genres)},
                )
            )
            .mappings()
            .one()
        )
        await conn.execute(text("UPDATE catalog_settings SET catalog_version=catalog_version+1"))
    return dict(row)


@app.patch("/items/{item_id}", tags=["Catalog administration"])
async def item_status(item_id: int, data: ItemStatus, user: User = Depends(admin_user)):
    async with engine.begin() as conn:
        await conn.execute(text("SELECT singleton FROM catalog_settings FOR UPDATE"))
        row = (
            (
                await conn.execute(
                    text("UPDATE items SET active=:active WHERE id=:id RETURNING *"),
                    {"active": data.active, "id": item_id},
                )
            )
            .mappings()
            .first()
        )
        if row is None:
            raise HTTPException(404, "Item not found")
        await conn.execute(text("UPDATE catalog_settings SET catalog_version=catalog_version+1"))
    return dict(row)


@app.get("/recommendations", tags=["Recommendations"])
async def recommendations(
    response: Response,
    limit: int = Query(default=10, ge=1, le=50),
    algorithm: Literal["auto", "popularity", "content", "collaborative", "hybrid"] = "auto",
    user: User = Depends(current_user),
):
    async with engine.connect() as base:
        conn = await base.execution_options(isolation_level="REPEATABLE READ")
        async with conn.begin():
            # Версии и данные читаются из одного снимка: кэш никогда не получает чужую версию истории.
            profile = (
                (
                    await conn.execute(
                        text("""
                SELECT profile_version, preferences FROM users WHERE id=:id
            """),
                        {"id": user.id},
                    )
                )
                .mappings()
                .one()
            )
            info, model = await active_model(conn)
            selected = model.manifest["default_algorithm"] if algorithm == "auto" else algorithm
            key = (
                user.id,
                profile["profile_version"],
                info["catalog_version"],
                info["version"],
                selected,
                limit,
            )
            if key in cache:
                cache.move_to_end(key)
                response.headers["X-Recommendation-Cache"] = "hit"
                return cache[key]
            state = (
                await conn.execute(
                    text("SELECT item_id, weight FROM user_item_state WHERE user_id=:user"),
                    {"user": user.id},
                )
            ).all()
            catalog = [
                dict(row)
                for row in (await conn.execute(text("SELECT * FROM items ORDER BY id"))).mappings()
            ]
    async with inference_slots:
        ranking = await asyncio.to_thread(
            model.rank, dict(state), catalog, profile["preferences"], selected, limit
        )
    result = {
        "model_version": info["version"],
        "profile_version": profile["profile_version"],
        "catalog_version": info["catalog_version"],
        "algorithm": selected,
        **ranking,
    }
    cache[key] = result
    cache.move_to_end(key)
    while len(cache) > 256:
        cache.popitem(last=False)
    response.headers["X-Recommendation-Cache"] = "miss"
    return result


@app.get("/models", tags=["Model administration"])
async def models(user: User = Depends(admin_user)):
    async with engine.connect() as conn:
        rows = (
            (
                await conn.execute(
                    text("""
            SELECT m.*, m.version=s.active_model AS active FROM models m CROSS JOIN catalog_settings s
            ORDER BY registered_at DESC
        """)
                )
            )
            .mappings()
            .all()
        )
    return [dict(row) for row in rows]


@app.post("/models/{version}/activate", tags=["Model administration"])
async def activate_model(version: str, user: User = Depends(admin_user)):
    from .config import settings
    from .recommendation import load_model

    async with engine.begin() as conn:
        await conn.execute(text("SELECT singleton FROM catalog_settings FOR UPDATE"))
        checksum = (
            await conn.execute(
                text("SELECT manifest_sha256 FROM models WHERE version=:version"),
                {"version": version},
            )
        ).scalar_one_or_none()
        if checksum is None:
            raise HTTPException(404, "Registered model not found")
        model = load_model(str(settings.model_dir), version, checksum)
        _, previous = await active_model(conn)
        if model.genres != previous.genres:
            raise HTTPException(409, "Genre taxonomy change requires an explicit profile migration")
        await conn.execute(
            text("UPDATE catalog_settings SET active_model=:version"), {"version": version}
        )
    return {"active_model": version}
