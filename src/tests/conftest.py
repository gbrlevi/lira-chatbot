import os
from pathlib import Path

import pytest
from fakes import FakeEmbedder

from app.config import REPO_ROOT
from app.ingest.run import build_store

# Tests never reach the model provider, even when a real key exists in the developer's .env.
os.environ["AI_API_KEY"] = ""
os.environ.pop("REFERENCE_DATETIME", None)

DATA_DIR = REPO_ROOT / "data"


@pytest.fixture(scope="session")
def embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture(scope="session")
def store_path(tmp_path_factory: pytest.TempPathFactory, embedder: FakeEmbedder) -> Path:
    path = tmp_path_factory.mktemp("store") / "emporio.db"
    build_store(DATA_DIR, path, embedder)
    return path
