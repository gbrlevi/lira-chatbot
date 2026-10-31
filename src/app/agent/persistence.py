"""Conversation memory on disk, so a customer can leave and come back to the same thread."""

import sqlite3
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver


def open_checkpointer(path: Path) -> SqliteSaver:
    """The API runs requests on several threads, and SqliteSaver serializes access with a lock."""
    path.parent.mkdir(parents=True, exist_ok=True)
    return SqliteSaver(sqlite3.connect(path, check_same_thread=False))
