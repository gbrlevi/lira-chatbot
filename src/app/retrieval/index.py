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
# Candidates taken from each ranking before merging.
CANDIDATES_PER_RANKING = 20
# Prefix length used as a light stemmer: "devolver" and "devolução" both become "devol".
STEM_LENGTH = 5

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
        "eu",
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
    similarity: float
    lexical_match: bool


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


def build_match_query(query: str) -> str | None:
    """Build a safe FTS5 expression. Quoting stops the customer's text acting as syntax."""
    terms = [term for term in normalize(query).split() if term not in _STOPWORDS]
    if not terms:
        return None
    unique = dict.fromkeys(term[:STEM_LENGTH] for term in terms)
    return " OR ".join(f'"{term}"*' for term in unique)


class HybridIndex:
    def __init__(self, store_path: Path, embedder: Embedder) -> None:
        self._store_path = store_path
        self._embedder = embedder
        self._vectors: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        with connect_readonly(store_path) as connection:
            built_with = connection.execute(
                "SELECT value FROM index_meta WHERE key = 'embedder'"
            ).fetchone()[0]
        if built_with != embedder.name:
            raise StaleIndexError(
                f"Índice criado com {built_with}, mas o provedor atual é {embedder.name}. "
                "Execute lira-ingest novamente."
            )

    def _load_vectors(self, connection: sqlite3.Connection, kind: str) -> tuple:
        if kind not in self._vectors:
            rows = connection.execute(
                "SELECT doc_id, vector FROM documents WHERE kind = ? ORDER BY doc_id", (kind,)
            ).fetchall()
            ids = np.array([row["doc_id"] for row in rows])
            matrix = np.stack([np.frombuffer(row["vector"], dtype=np.float32) for row in rows])
            self._vectors[kind] = (ids, matrix)
        return self._vectors[kind]

    @staticmethod
    def _lexical_ranking(connection: sqlite3.Connection, query: str, kind: str) -> list[int]:
        expression = build_match_query(query)
        if expression is None:
            return []
        rows = connection.execute(
            "SELECT rowid FROM documents_fts WHERE documents_fts MATCH ? AND kind = ?"
            " ORDER BY bm25(documents_fts) LIMIT ?",
            (expression, kind, CANDIDATES_PER_RANKING),
        ).fetchall()
        return [row["rowid"] for row in rows]

    def search(
        self,
        query: str,
        kind: str,
        k: int = 4,
        min_similarity: float = 0.0,
        allowed_refs: set[str] | None = None,
    ) -> list[SearchHit]:
        """Return up to k documents.

        A document qualifies through a lexical match or a similarity of at least
        min_similarity, so an irrelevant query can return nothing instead of its nearest
        neighbor. allowed_refs restricts results to refs that passed hard filters.
        """
        with connect_readonly(self._store_path) as connection:
            ids, matrix = self._load_vectors(connection, kind)
            similarities = matrix @ self._embedder.embed_query(query)
            dense_order = np.argsort(-similarities)[:CANDIDATES_PER_RANKING]
            similarity_by_id = dict(zip(ids.tolist(), similarities.tolist(), strict=True))
            lexical = self._lexical_ranking(connection, query, kind)

            scores: dict[int, float] = {}
            for rank, position in enumerate(dense_order, start=1):
                doc_id = int(ids[position])
                if similarity_by_id[doc_id] >= min_similarity:
                    scores[doc_id] = 1 / (RRF_K + rank)
            for rank, doc_id in enumerate(lexical, start=1):
                scores[doc_id] = scores.get(doc_id, 0.0) + 1 / (RRF_K + rank)

            hits: list[SearchHit] = []
            lexical_ids = set(lexical)
            for doc_id in sorted(scores, key=lambda key: -scores[key]):
                row = connection.execute(
                    "SELECT ref, title, text FROM documents WHERE doc_id = ?", (doc_id,)
                ).fetchone()
                if allowed_refs is not None and row["ref"] not in allowed_refs:
                    continue
                hits.append(
                    SearchHit(
                        ref=row["ref"],
                        title=row["title"],
                        text=row["text"],
                        score=scores[doc_id],
                        similarity=similarity_by_id[doc_id],
                        lexical_match=doc_id in lexical_ids,
                    )
                )
                if len(hits) == k:
                    break
        return hits
