"""Read local configuration without putting passwords in connection strings."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy import URL

PROJECT_ROOT = Path(__file__).resolve().parents[2]
VECTOR_DIMENSIONS = 1024


class ConfigurationError(ValueError):
    """A required database setting is missing."""


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    db_host: str = ""
    db_port: int = Field(default=5432, ge=1, le=65535)
    db_name: str = "postgres"
    db_user: str = ""
    db_password: SecretStr = SecretStr("")
    db_sslmode: Literal["require", "verify-ca", "verify-full"] = "require"
    db_connect_timeout: int = Field(default=10, ge=1, le=60)
    embedding_model: str = "BAAI/bge-m3"

    def database_url(self) -> URL:
        missing = []
        for name in ("db_host", "db_user", "db_password"):
            value = getattr(self, name)
            if isinstance(value, SecretStr):
                value = value.get_secret_value()
            if not str(value).strip():
                missing.append(name.upper())
        if missing:
            raise ConfigurationError("请在本机 .env 填写：" + ", ".join(missing))
        return URL.create(
            "postgresql+psycopg",
            username=self.db_user,
            password=self.db_password.get_secret_value(),
            host=self.db_host,
            port=self.db_port,
            database=self.db_name,
        )


@lru_cache
def get_settings() -> Settings:
    return Settings()

