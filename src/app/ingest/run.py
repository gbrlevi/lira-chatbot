"""Rebuilds the store database from data/. Safe to run repeatedly."""

import os
from dataclasses import dataclass
from pathlib import Path

from app.config import get_settings
from app.db import connect_writable
from app.ingest.catalog import CatalogReport, load_catalog
from app.ingest.documents import policy_documents, product_documents
from app.ingest.policies import chunk_markdown, extract_markdown
from app.retrieval.embeddings import Embedder, get_embedder
from app.retrieval.index import write_index

POLICY_PDF_NAME = "políticas_da_loja.pdf"


@dataclass(frozen=True)
class IngestReport:
    catalog: CatalogReport
    policy_chunks: int
    embedder: str


def build_store(data_dir: Path, target: Path, embedder: Embedder) -> IngestReport:
    """Build into a temporary file and swap it in, so a failure keeps the previous store."""
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".tmp")
    temporary.unlink(missing_ok=True)
    try:
        with connect_writable(temporary) as connection:
            catalog = load_catalog(connection, data_dir)
            chunks = chunk_markdown(extract_markdown(data_dir / POLICY_PDF_NAME))
            documents = [*policy_documents(chunks), *product_documents(connection)]
            write_index(connection, documents, embedder)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return IngestReport(catalog=catalog, policy_chunks=len(chunks), embedder=embedder.name)


def main() -> None:
    settings = get_settings()
    report = build_store(settings.data_dir, settings.store_db_path, get_embedder())
    catalog = report.catalog
    print(f"Base criada em {settings.store_db_path}")
    print(f"  Embeddings: {report.embedder}")
    print(f"  Produtos: {catalog.products}")
    print(f"  Trechos do manual: {report.policy_chunks}")
    print(f"  Descrições desconsideradas por conflito: {catalog.unreliable_descriptions}")
    print(f"  Pedidos com total diferente da soma dos itens: {catalog.order_total_mismatches}")
    print(f"  E-mails compartilhados entre clientes: {catalog.shared_emails}")


if __name__ == "__main__":
    main()
