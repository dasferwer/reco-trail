from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    database_url: str = "postgresql+asyncpg://recotrail:recotrail@localhost:5432/recotrail"
    jwt_secret: str = Field(min_length=32)
    token_minutes: int = Field(default=60, ge=1, le=1440)
    model_dir: Path = Path("models")


@lru_cache
def get_settings():
    return Settings()


settings = get_settings()
