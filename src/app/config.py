"""Application settings, read from environment variables and the local .env file."""

from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=REPO_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    ai_api_key: str = Field(default="", alias="AI_API_KEY")
    llm_model: str = Field(default="gemini-3.5-flash-lite", alias="LLM_MODEL")
    llm_timeout_seconds: float = Field(default=30.0, alias="LLM_TIMEOUT_SECONDS")
    embeddings_provider: Literal["fastembed", "gemini"] = Field(
        default="fastembed", alias="EMBEDDINGS_PROVIDER"
    )
    reference_datetime: datetime | None = Field(default=None, alias="REFERENCE_DATETIME")

    data_dir: Path = Field(default=REPO_ROOT / "data", alias="DATA_DIR")
    var_dir: Path = Field(default=REPO_ROOT / "var", alias="VAR_DIR")

    @property
    def prompts_dir(self) -> Path:
        return SRC_DIR / "prompts"

    @property
    def store_db_path(self) -> Path:
        """Catalog and policy index. Rebuilt from data/ by the ingestion command."""
        return self.var_dir / "emporio.db"

    @property
    def runtime_db_path(self) -> Path:
        """Conversation checkpoints and tickets. Survives re-ingestion."""
        return self.var_dir / "runtime.db"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
