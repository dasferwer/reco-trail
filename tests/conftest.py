import json
import os
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import UUID

import httpx
import jwt
import pytest
import pytest_asyncio
from sqlalchemy import text

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from recotrail.config import settings  # noqa: E402
from recotrail.db import engine  # noqa: E402
from recotrail.main import app, cache  # noqa: E402
from recotrail.recommendation import Model, digest, load_model  # noqa: E402

USER = UUID("16000000-0000-0000-0000-000000000010")
OTHER = UUID("16000000-0000-0000-0000-000000000011")
ADMIN = UUID("16000000-0000-0000-0000-000000000012")


@pytest.fixture(scope="session")
def model():
    return Model(ROOT / "models/movielens-v1")


@pytest_asyncio.fixture(autouse=True)
async def database(model):
    if os.environ.get("TESTING") != "true" or not settings.database_url.endswith("/recotrail_test"):
        raise RuntimeError("Tests require the isolated recotrail_test database")
    cache.clear()
    load_model.cache_clear()
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "TRUNCATE users, items, events, user_item_state, models, catalog_settings RESTART IDENTITY CASCADE"
            )
        )
        for user_id, role in [(USER, "user"), (OTHER, "user"), (ADMIN, "admin")]:
            await conn.execute(
                text(
                    "INSERT INTO users(id,email,password_hash,role) VALUES (:id,:email,'unused',:role)"
                ),
                {"id": user_id, "email": str(user_id) + "@example.com", "role": role},
            )
        raw = await conn.get_raw_connection()
        await raw.driver_connection.copy_records_to_table(
            "items",
            columns=["id", "title", "genres"],
            records=[(i["id"], i["title"], json.dumps(i["genres"])) for i in model.catalog],
        )
        await conn.execute(
            text("INSERT INTO models VALUES ('movielens-v1', :hash, '{}', now())"),
            {"hash": digest(ROOT / "models/movielens-v1/manifest.json")},
        )
        await conn.execute(
            text("INSERT INTO catalog_settings(active_model) VALUES ('movielens-v1')")
        )
    yield


def headers(user_id=USER):
    now = datetime.now(UTC)
    token = jwt.encode(
        {
            "sub": str(user_id),
            "iat": now,
            "exp": now + timedelta(minutes=5),
            "iss": "recotrail",
            "aud": "recotrail",
        },
        settings.jwt_secret,
        algorithm="HS256",
    )
    return {"Authorization": "Bearer " + token}


@pytest_asyncio.fixture
async def client():
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test", headers=headers()
    ) as client:
        yield client
