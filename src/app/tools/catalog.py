"""Catalog queries. Hard filters decide what exists; semantic search only orders the result."""

import json
import sqlite3
from typing import Literal

from langchain_core.tools import BaseTool, tool
from pydantic import BaseModel, Field
from rapidfuzz import fuzz

from app.db import connect_readonly
from app.domain.availability import Availability, classify
from app.domain.pricing import compute_price, format_brl
from app.ingest.documents import KIND_PRODUCT
from app.text import compact, normalize
from app.tools.context import ToolContext, to_json

Category = Literal[
    "Guitarras",
    "Baixos",
    "Baterias e Percussão",
    "Teclados e Pianos",
    "Violões",
    "Instrumentos de Sopro (Madeiras)",
    "Instrumentos de Sopro (Metais)",
    "Cordas Orquestrais",
    "Ukuleles",
]
SortOrder = Literal["relevancia", "menor_preco", "maior_preco", "mais_recentes"]

MAX_RESULTS = 30
MAX_ALTERNATIVES = 3
# Below this, a misspelled model is not treated as the product the customer meant.
FUZZY_MODEL_THRESHOLD = 85
# Words naming the instrument type, which say nothing about the model being asked for.
_GENERIC_WORDS = frozenset(
    [
        "violao",
        "violoes",
        "guitarra",
        "guitarras",
        "baixo",
        "baixos",
        "ukulele",
        "ukuleles",
        "teclado",
        "teclados",
        "bateria",
        "baterias",
        "instrumento",
        "modelo",
        "marca",
    ]
)
_SPEC_LABELS = {
    "top": "tampo",
    "back_sides": "fundo e laterais",
    "neck": "braço",
    "scale": "escala",
    "body": "corpo",
    "shells": "cascos",
    "pieces": "peças",
    "keys": "teclas",
    "polyphony": "polifonia",
}

_PRODUCT_QUERY = """
SELECT p.*, c.name AS category,
       pr.discount_percent AS promotion_percent,
       pr.description AS promotion_name
FROM products p
JOIN categories c ON c.category_id = p.category_id
LEFT JOIN promotions pr ON pr.product_id = p.product_id AND pr.is_active = 1
"""


def _availability(row: sqlite3.Row) -> Availability:
    return classify(row["status"], row["stock_quantity"])


def _present(row: sqlite3.Row, *, details: bool = False) -> dict:
    """Shape a product for the agent. Stock counts are never included."""
    availability = _availability(row)
    product: dict = {
        "id": row["product_id"],
        "nome": row["name"],
        "marca": row["brand"],
        "categoria": row["category"],
        "disponibilidade": availability.value,
    }
    # Showing a price for what cannot be bought would read as an offer.
    if availability in (Availability.DISCONTINUED, Availability.COMING_SOON):
        product["preco"] = "não informado para este produto"
    else:
        price = compute_price(row["price_cents"], row["promotion_percent"], row["promotion_name"])
        if price.discount_percent:
            product["preco_original"] = format_brl(price.original_cents)
            product["desconto"] = f"{price.discount_percent}%"
            product["preco_promocional"] = format_brl(price.final_cents)
            product["campanha"] = price.promotion_name
        else:
            product["preco"] = format_brl(price.final_cents)
            product["preco_no_pix"] = format_brl(price.pix_cents)
    if row["string_type"]:
        product["cordas"] = row["string_type"]
    if row["string_count"]:
        product["numero_de_cordas"] = row["string_count"]
    if row["has_electronics"] is not None:
        product["eletrico"] = bool(row["has_electronics"])
    if row["color"]:
        product["cor"] = row["color"]
    if details:
        specs = json.loads(row["specs"])
        product["especificacoes"] = {
            label: specs[key] for key, label in _SPEC_LABELS.items() if key in specs
        }
        if row["description_reliable"]:
            product["descricao"] = row["description"]
    return product


def _match_model(rows: list[sqlite3.Row], query: str) -> tuple[list[sqlite3.Row], bool]:
    """Find products by model or brand. Returns the matches and whether they are approximate.

    Fuzzy matching only runs when nothing matches exactly, otherwise "Yamaha C40" would
    also bring the C70, which differs by a single character.
    """
    terms = [term for term in normalize(query).split() if term not in _GENERIC_WORDS]
    if not terms:
        return [], False

    def contains_all(row: sqlite3.Row) -> bool:
        return all(compact(term) in row["search_key"] for term in terms)

    exact = [row for row in rows if contains_all(row)]
    if exact:
        return exact, False

    wanted = compact(" ".join(terms))
    scored = [(fuzz.partial_ratio(wanted, row["search_key"]), row) for row in rows]
    close = [(score, row) for score, row in scored if score >= FUZZY_MODEL_THRESHOLD]
    close.sort(key=lambda pair: -pair[0])
    return [row for _, row in close], True


def _alternatives(
    connection: sqlite3.Connection, row: sqlite3.Row, excluded: set[int]
) -> list[dict]:
    """Purchasable products of the same category, closest in price first (policy 7.3)."""
    candidates = connection.execute(
        f"{_PRODUCT_QUERY} WHERE p.category_id = ? AND p.status = 'active'"
        " AND p.stock_quantity > 0 ORDER BY ABS(p.price_cents - ?)",
        (row["category_id"], row["price_cents"]),
    ).fetchall()
    chosen = [c for c in candidates if c["product_id"] not in excluded][:MAX_ALTERNATIVES]
    return [_present(candidate) for candidate in chosen]


def _order_by(rows: list[sqlite3.Row], order: SortOrder) -> list[sqlite3.Row]:
    def final_price(row: sqlite3.Row) -> int:
        return compute_price(row["price_cents"], row["promotion_percent"]).final_cents

    if order == "menor_preco":
        return sorted(rows, key=final_price)
    if order == "maior_preco":
        return sorted(rows, key=final_price, reverse=True)
    if order == "mais_recentes":
        return sorted(rows, key=lambda row: row["created_at"], reverse=True)
    return rows


def search_products(
    context: ToolContext,
    *,
    modelo: str | None = None,
    caracteristica: str | None = None,
    categoria: Category | None = None,
    marca: str | None = None,
    preco_min: float | None = None,
    preco_max: float | None = None,
    cordas: Literal["nylon", "aço"] | None = None,
    numero_de_cordas: int | None = None,
    eletrico: bool | None = None,
    cor: str | None = None,
    apenas_promocao: bool = False,
    incluir_indisponiveis: bool = False,
    ordenar: SortOrder = "relevancia",
    limite: int = 15,
) -> dict:
    with connect_readonly(context.store_path) as connection:
        rows = connection.execute(f"{_PRODUCT_QUERY} ORDER BY p.product_id").fetchall()

        def keep(row: sqlite3.Row) -> bool:
            final = compute_price(row["price_cents"], row["promotion_percent"]).final_cents
            conditions = (
                categoria is None or row["category"] == categoria,
                marca is None or normalize(row["brand"]) == normalize(marca),
                preco_min is None or final >= round(preco_min * 100),
                preco_max is None or final <= round(preco_max * 100),
                cordas is None or row["string_type"] == cordas,
                numero_de_cordas is None or row["string_count"] == numero_de_cordas,
                eletrico is None or bool(row["has_electronics"]) is eletrico,
                cor is None or row["color_family"] == normalize(cor),
                not apenas_promocao or row["promotion_percent"] is not None,
            )
            return all(conditions)

        rows = [row for row in rows if keep(row)]

        # A customer asking for a model must hear when it is unavailable, not that it is
        # missing, so naming a model always includes unavailable products.
        approximate = False
        if modelo:
            rows, approximate = _match_model(rows, modelo)
        elif not incluir_indisponiveis:
            rows = [row for row in rows if _availability(row).purchasable]

        if caracteristica:
            # Without a category, similarity would suggest a ukulele for "saxofone", so the
            # words themselves must appear in the product.
            if categoria is None:
                literal = context.index.matching_refs(caracteristica, KIND_PRODUCT)
                rows = [row for row in rows if str(row["product_id"]) in literal]
            allowed = {str(row["product_id"]) for row in rows}
            if ordenar == "relevancia" and allowed:
                hits = context.index.search(
                    caracteristica, KIND_PRODUCT, k=len(allowed), allowed_refs=allowed
                )
                by_id = {str(row["product_id"]): row for row in rows}
                rows = [by_id[hit.ref] for hit in hits]

        if ordenar == "relevancia" and not caracteristica and not modelo:
            ordenar = "menor_preco"
        rows = _order_by(rows, ordenar)

        total = len(rows)
        shown = rows[: min(limite, MAX_RESULTS)]
        products = [_present(row) for row in shown]

        # Few results mean a specific question, where alternatives are what the policy asks for.
        if total <= MAX_ALTERNATIVES:
            shown_ids = {row["product_id"] for row in shown}
            for row, product in zip(shown, products, strict=True):
                if not _availability(row).purchasable:
                    product["alternativas"] = _alternatives(connection, row, shown_ids)

    result: dict = {"total_encontrado": total, "produtos": products}
    if total > len(shown):
        result["observacao"] = f"Mostrando {len(shown)} de {total}. Refine a busca se precisar."
    if approximate and products:
        result["observacao"] = (
            "Nenhum modelo com esse nome exato. Estes têm nome parecido: confirme com o "
            "cliente antes de tratá-los como o produto pedido."
        )
    if not total:
        result["observacao"] = (
            "Nada encontrado no sistema com esses critérios. Diga isso ao cliente sem afirmar "
            "que a loja nunca trabalhou com o item."
        )
    return result


def get_product_details(context: ToolContext, ids: list[int]) -> dict:
    with connect_readonly(context.store_path) as connection:
        placeholders = ", ".join("?" * len(ids))
        rows = connection.execute(
            f"{_PRODUCT_QUERY} WHERE p.product_id IN ({placeholders})", ids
        ).fetchall()
        products = []
        for row in rows:
            product = _present(row, details=True)
            if not _availability(row).purchasable:
                product["alternativas"] = _alternatives(connection, row, set(ids))
            products.append(product)
    missing = sorted(set(ids) - {row["product_id"] for row in rows})
    result: dict = {"produtos": products}
    if missing:
        result["ids_nao_encontrados"] = missing
    return result


def summarize_catalog(context: ToolContext, categoria: Category | None = None) -> dict:
    with connect_readonly(context.store_path) as connection:
        categories = connection.execute(
            "SELECT name FROM categories WHERE ? IS NULL OR name = ? ORDER BY category_id",
            (categoria, categoria),
        ).fetchall()
        rows = connection.execute(_PRODUCT_QUERY).fetchall()

    summary = []
    for category in categories:
        name = category["name"]
        available = [r for r in rows if r["category"] == name and _availability(r).purchasable]
        entry: dict = {"categoria": name, "disponiveis": len(available)}
        if available:
            prices = [
                compute_price(r["price_cents"], r["promotion_percent"]).final_cents
                for r in available
            ]
            entry["menor_preco"] = format_brl(min(prices))
            entry["maior_preco"] = format_brl(max(prices))
            entry["marcas"] = sorted({r["brand"] for r in available})
        summary.append(entry)
    return {"categorias": summary}


class SearchProductsInput(BaseModel):
    modelo: str | None = Field(
        default=None,
        description=(
            "Modelo ou marca citados pelo cliente, como 'Takamine GD20'. Traz também produtos "
            "indisponíveis, para você informar a situação deles."
        ),
    )
    caracteristica: str | None = Field(
        default=None,
        description=(
            "Uso ou característica desejados, como 'para iniciante'. Informe também a "
            "categoria sempre que souber."
        ),
    )
    categoria: Category | None = Field(default=None, description="Categoria do instrumento.")
    marca: str | None = Field(default=None, description="Marca exata, como 'Yamaha'.")
    preco_min: float | None = Field(default=None, ge=0, description="Preço mínimo em reais.")
    preco_max: float | None = Field(default=None, ge=0, description="Preço máximo em reais.")
    cordas: Literal["nylon", "aço"] | None = Field(default=None, description="Tipo de corda.")
    numero_de_cordas: int | None = Field(default=None, description="Número de cordas, como 7.")
    eletrico: bool | None = Field(
        default=None, description="true para modelos com captação ou eletroacústicos."
    )
    cor: str | None = Field(default=None, description="Cor em português, como 'preto'.")
    apenas_promocao: bool = Field(
        default=False, description="true para listar só o que tem promoção ativa."
    )
    incluir_indisponiveis: bool = Field(
        default=False, description="true para listar também o que não pode ser comprado agora."
    )
    ordenar: SortOrder = Field(
        default="relevancia", description="Ordem dos resultados, como 'menor_preco'."
    )
    limite: int = Field(default=15, ge=1, le=MAX_RESULTS, description="Máximo de resultados.")


class ProductDetailsInput(BaseModel):
    ids: list[int] = Field(
        min_length=1, max_length=5, description="IDs dos produtos, vindos de buscar_produtos."
    )


class CatalogSummaryInput(BaseModel):
    categoria: Category | None = Field(
        default=None, description="Categoria a resumir. Omita para ver todas."
    )


def build_catalog_tools(context: ToolContext) -> list[BaseTool]:
    @tool("buscar_produtos", args_schema=SearchProductsInput)
    def buscar_produtos(**filters) -> str:
        """Busca instrumentos e retorna preço, promoção ativa e disponibilidade atuais.

        Use em toda pergunta sobre produto, preço, estoque, promoção ou recomendação; nunca
        responda esses temas de memória. Por padrão lista só o que pode ser comprado. Os
        preços já vêm calculados: repita-os exatamente, sem fazer contas.
        """
        return to_json(search_products(context, **filters))

    @tool("detalhar_produtos", args_schema=ProductDetailsInput)
    def detalhar_produtos(ids: list[int]) -> str:
        """Retorna especificações e descrição de até 5 produtos, para detalhar ou comparar.

        Use depois de buscar_produtos, quando o cliente quiser saber mais. Só afirme
        características que constarem no retorno.
        """
        return to_json(get_product_details(context, ids))

    @tool("resumir_catalogo", args_schema=CatalogSummaryInput)
    def resumir_catalogo(categoria: Category | None = None) -> str:
        """Retorna, por categoria, quantos modelos estão disponíveis, faixa de preço e marcas.

        Use em perguntas gerais como 'o que vocês vendem?' ou 'quais marcas de violão têm?'.
        """
        return to_json(summarize_catalog(context, categoria))

    return [buscar_produtos, detalhar_produtos, resumir_catalogo]
