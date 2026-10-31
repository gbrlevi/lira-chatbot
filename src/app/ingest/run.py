"""Rebuilds the store database from data/. Safe to run repeatedly."""

import os
from pathlib import Path

from app.config import get_settings
from app.db import connect_writable
from app.ingest.catalog import CatalogReport, load_catalog


def build_store(data_dir: Path, target: Path) -> CatalogReport:
    """Build into a temporary file and swap it in, so a failure keeps the previous store."""
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".tmp")
    temporary.unlink(missing_ok=True)
    try:
        with connect_writable(temporary) as connection:
            report = load_catalog(connection, data_dir)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return report


def main() -> None:
    settings = get_settings()
    report = build_store(settings.data_dir, settings.store_db_path)
    print(f"Base criada em {settings.store_db_path}")
    print(f"  Produtos: {report.products}")
    print(f"  Descrições desconsideradas por conflito: {report.unreliable_descriptions}")
    print(f"  Pedidos com total diferente da soma dos itens: {report.order_total_mismatches}")
    print(f"  E-mails compartilhados entre clientes: {report.shared_emails}")


if __name__ == "__main__":
    main()
