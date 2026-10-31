"""Builds the searchable documents for policies and products."""

import sqlite3

from app.ingest.policies import PolicyChunk
from app.retrieval.index import Document

KIND_POLICY = "policy"
KIND_PRODUCT = "product"


def policy_documents(chunks: list[PolicyChunk]) -> list[Document]:
    return [
        Document(
            kind=KIND_POLICY,
            ref=chunk.section,
            title=chunk.title,
            text=chunk.searchable_text,
        )
        for chunk in chunks
    ]


def product_documents(connection: sqlite3.Connection) -> list[Document]:
    """Descriptions that contradict the product stay out of the semantic index."""
    rows = connection.execute(
        """
        SELECT p.product_id, p.name, p.description, p.description_reliable,
               p.string_type, p.color, c.name AS category
        FROM products p JOIN categories c ON c.category_id = p.category_id
        ORDER BY p.product_id
        """
    ).fetchall()
    documents = []
    for row in rows:
        parts = [row["name"], row["category"]]
        if row["string_type"]:
            parts.append(f"cordas de {row['string_type']}")
        if row["color"]:
            parts.append(f"cor {row['color']}")
        if row["description_reliable"]:
            parts.append(row["description"])
        documents.append(
            Document(
                kind=KIND_PRODUCT,
                ref=str(row["product_id"]),
                title=row["name"],
                text=". ".join(parts),
            )
        )
    return documents
