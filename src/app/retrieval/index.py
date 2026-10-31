"""Hybrid retrieval: BM25 (SQLite FTS5) and exact cosine (NumPy), merged with RRF.

BM25 finds exact terms such as PIX or Jadlog. Vectors find paraphrases such as "me
arrependi". A brute-force scan is enough for the few dozen documents of this PoC.
"""

import sqlite3
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.db import connect_readonly
from app.retrieval.embeddings import Embedder
from app.text import normalize

RRF_K = 60
# Prefix length used as a light stemmer: "devolver" and "devolução" both become "devol".
STEM_LENGTH = 5
# Words shorter than this are matched whole by the strict word match.
STRICT_STEM_MIN_LENGTH = 6

SCHEMA = """
CREATE TABLE index_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE documents (
    doc_id INTEGER PRIMARY KEY,
    kind TEXT NOT NULL,
    ref TEXT NOT NULL,
    title TEXT NOT NULL,
    text TEXT NOT NULL,
    vector BLOB NOT NULL
);
CREATE INDEX idx_documents_kind ON documents (kind);
CREATE VIRTUAL TABLE documents_fts USING fts5(
    content,
    kind UNINDEXED,
    tokenize = "unicode61 remove_diacritics 2"
);
"""

_STOPWORDS = frozenset(
    [
        "a",
        "ao",
        "aos",
        "as",
        "com",
        "como",
        "da",
        "das",
        "de",
        "do",
        "dos",
        "e",
        "em",
        "eu",
        "essa",
        "esse",
        "esta",
        "este",
        "ha",
        "isso",
        "me",
        "meu",
        "minha",
        "na",
        "nas",
        "no",
        "nos",
        "nossa",
        "nosso",
        "o",
        "os",
        "ou",
        "para",
        "pela",
        "pelo",
        "por",
        "pra",
        "pro",
        "qual",
        "quais",
        "quando",
        "quanto",
        "que",
        "se",
        "sem",
        "ser",
        "sobre",
        "sua",
        "seu",
        "tem",
        "ter",
        "um",
        "uma",
        "umas",
        "uns",
        "voce",
        "voces",
        "vcs",
    ]
)


class StaleIndexError(RuntimeError):
    """The index was built with a different embedder and must be rebuilt."""


@dataclass(frozen=True)
class Document:
    kind: str
    ref: str
    title: str
    text: str


@dataclass(frozen=True)
class SearchHit:
    ref: str
    title: str
    text: str
    score: float


def write_index(
    connection: sqlite3.Connection, documents: list[Document], embedder: Embedder
) -> None:
    connection.executescript(SCHEMA)
    connection.execute("INSERT INTO index_meta VALUES ('embedder', ?)", (embedder.name,))
    vectors = embedder.embed_documents([document.text for document in documents])
    for doc_id, (document, vector) in enumerate(zip(documents, vectors, strict=True), start=1):
        connection.execute(
            "INSERT INTO documents VALUES (?, ?, ?, ?, ?, ?)",
            (
                doc_id,
                document.kind,
                document.ref,
                document.title,
                document.text,
                vector.astype(np.float32).tobytes(),
            ),
        )
        connection.execute(
            "INSERT INTO documents_fts (rowid, content, kind) VALUES (?, ?, ?)",
            (doc_id, normalize(document.text), document.kind),
        )


def _terms(query: str) -> list[str]:
    unique = dict.fromkeys(normalize(query).split())
    return [term for term in unique if term not in _STOPWORDS]


def build_match_query(query: str) -> str | None:
    """Loose expression used for ranking. Quoting stops customer text acting as FTS syntax."""
    stems = dict.fromkeys(term[:STEM_LENGTH] for term in _terms(query))
    return " OR ".join(f'"{stem}"*' for stem in stems) or None


def build_strict_match_query(query: str) -> str | None:
    """Expression used to decide existence: every word must appear, allowing plural forms."""
    stems = (term[:-2] if len(term) >= STRICT_STEM_MIN_LENGTH else term for term in _terms(query))
    return " AND ".join(f'"{stem}"*' for stem in stems) or None


class HybridIndex:
    def __init__(self, store_path: Path, embedder: Embedder) -> None:
        self._store_path = store_path
        self._embedder = embedder
        self._vectors: dict[str, tuple[list[int], np.ndarray]] = {}
        with connect_readonly(store_path) as connection:
            built_with = connection.execute(
                "SELECT value FROM index_meta WHERE key = 'embedder'"
            ).fetchone()[0]
        if built_with != embedder.name:
            raise StaleIndexError(
                f"Índice criado com {built_with}, mas o provedor atual é {embedder.name}. "
                "Execute lira-ingest novamente."
            )

    def _load_vectors(
        self, connection: sqlite3.Connection, kind: str
    ) -> tuple[list[int], np.ndarray]:
        if kind not in self._vectors:
            rows = connection.execute(
                "SELECT doc_id, vector FROM documents WHERE kind = ? ORDER BY doc_id", (kind,)
            ).fetchall()
            ids = [row["doc_id"] for row in rows]
            matrix = np.stack([np.frombuffer(row["vector"], dtype=np.float32) for row in rows])
            self._vectors[kind] = (ids, matrix)
        return self._vectors[kind]

    def matching_refs(self, query: str, kind: str) -> set[str]:
        """Refs of the documents that literally contain every word of the query."""
        expression = build_strict_match_query(query)
        if expression is None:
            return set()
        with connect_readonly(self._store_path) as connection:
            rows = connection.execute(
                """
                SELECT d.ref FROM documents_fts f JOIN documents d ON d.doc_id = f.rowid
                WHERE documents_fts MATCH ? AND f.kind = ?
                """,
                (expression, kind),
            ).fetchall()
        return {row["ref"] for row in rows}

    def search(
        self, query: str, kind: str, k: int = 4, allowed_refs: set[str] | None = None
    ) -> list[SearchHit]:
        """Rank documents by relevance and return the best k.

        This only orders documents. Cosine similarity does not separate relevant from
        irrelevant here (measured: "trompete" scores above "baixo para banda"), so callers
        decide what exists through hard filters passed in allowed_refs.
        """
        with connect_readonly(self._store_path) as connection:
            ids, matrix = self._load_vectors(connection, kind)
            similarities = matrix @ self._embedder.embed_query(query)
            scores: dict[int, float] = {}
            for rank, position in enumerate(np.argsort(-similarities), start=1):
                scores[ids[int(position)]] = 1 / (RRF_K + rank)

            expression = build_match_query(query)
            if expression:
                rows = connection.execute(
                    "SELECT rowid FROM documents_fts WHERE documents_fts MATCH ? AND kind = ?"
                    " ORDER BY bm25(documents_fts)",
                    (expression, kind),
                ).fetchall()
                for rank, row in enumerate(rows, start=1):
                    scores[row["rowid"]] += 1 / (RRF_K + rank)

            hits: list[SearchHit] = []
            for doc_id in sorted(scores, key=lambda key: -scores[key]):
                row = connection.execute(
                    "SELECT ref, title, text FROM documents WHERE doc_id = ?", (doc_id,)
                ).fetchone()
                if allowed_refs is not None and row["ref"] not in allowed_refs:
                    continue
                hits.append(SearchHit(row["ref"], row["title"], row["text"], scores[doc_id]))
                if len(hits) == k:
                    break
        return hits
