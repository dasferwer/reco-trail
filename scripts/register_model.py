"""Регистрируем подготовленную локальную версию; переключение выполняется отдельно через API."""

import argparse
import asyncio
import json

from sqlalchemy import text

from recotrail.config import settings
from recotrail.db import engine
from recotrail.recommendation import digest, load_model


async def register(version):
    import re

    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", version):
        raise ValueError("Invalid version")
    directory = settings.model_dir / version
    checksum = digest(directory / "manifest.json")
    model = load_model(str(settings.model_dir), version, checksum)
    if model.manifest.get("version") != version or model.manifest.get("default_algorithm") not in {
        "popularity",
        "content",
        "collaborative",
        "hybrid",
    }:
        raise ValueError("Invalid serving metadata")
    evaluation = json.loads((directory / "evaluation.json").read_text())
    async with engine.begin() as conn:
        await conn.execute(
            text("""
            INSERT INTO models(version, manifest_sha256, evaluation)
            VALUES (:version, :hash, CAST(:evaluation AS jsonb)) ON CONFLICT DO NOTHING
        """),
            {"version": version, "hash": checksum, "evaluation": json.dumps(evaluation)},
        )
        registered = (
            await conn.execute(
                text("SELECT manifest_sha256 FROM models WHERE version=:v"), {"v": version}
            )
        ).scalar_one()
        if registered != checksum:
            raise ValueError("Version already registered with different contents")
    print(json.dumps({"registered": version, "manifest_sha256": checksum}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("version")
    asyncio.run(register(parser.parse_args().version))
