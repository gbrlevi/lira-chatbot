"""The single place where tools are registered."""

from langchain_core.tools import BaseTool

from app.config import get_settings
from app.retrieval.embeddings import get_embedder
from app.retrieval.index import HybridIndex
from app.runtime_store import RuntimeStore
from app.tools.catalog import build_catalog_tools
from app.tools.context import ToolContext
from app.tools.orders import build_order_tools
from app.tools.service import build_service_tools


def build_tools(context: ToolContext) -> list[BaseTool]:
    return [
        *build_catalog_tools(context),
        *build_order_tools(context),
        *build_service_tools(context),
    ]


def default_context() -> ToolContext:
    settings = get_settings()
    return ToolContext(
        store_path=settings.store_db_path,
        index=HybridIndex(settings.store_db_path, get_embedder()),
        runtime=RuntimeStore(settings.runtime_db_path),
    )
