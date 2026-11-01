import json
from pathlib import Path

import pytest
from fakes import FakeEmbedder

from app.retrieval.index import HybridIndex
from app.runtime_store import RuntimeStore
from app.tools.catalog import (
    build_catalog_tools,
    get_product_details,
    search_products,
    summarize_catalog,
)
from app.tools.context import ToolContext


@pytest.fixture(scope="module")
def context(
    store_path: Path, embedder: FakeEmbedder, tmp_path_factory: pytest.TempPathFactory
) -> ToolContext:
    runtime = RuntimeStore(tmp_path_factory.mktemp("runtime") / "runtime.db")
    return ToolContext(store_path, HybridIndex(store_path, embedder), runtime)


def _ids(result: dict) -> set[int]:
    return {product["id"] for product in result["produtos"]}


def test_acoustic_guitars_up_to_1000_are_exactly_twelve(context: ToolContext) -> None:
    result = search_products(context, categoria="Violões", preco_max=1000, limite=30)
    assert _ids(result) == {81, 82, 83, 84, 85, 86, 87, 88, 89, 91, 107, 112}
    assert result["total_encontrado"] == 12


def test_unavailable_products_stay_out_of_listings(context: ToolContext) -> None:
    ids = _ids(search_products(context, categoria="Violões", preco_max=1000, limite=30))
    assert 96 not in ids  # active, but out of stock
    assert 113 not in ids  # discontinued


def test_price_filter_uses_the_promotional_price(context: ToolContext) -> None:
    # Ohana CK-20 costs R$ 549,00, but R$ 439,20 with its active 20% promotion.
    ids = _ids(search_products(context, categoria="Ukuleles", preco_max=440))
    assert 121 in ids


@pytest.mark.parametrize(
    "query", ["Takamine GD20", "takamine gd-20", "TAKAMINE GD 20", "GD20", "violão Takamine GD20"]
)
def test_model_search_ignores_format_and_generic_words(context: ToolContext, query: str) -> None:
    assert _ids(search_products(context, modelo=query)) == {95}


def test_model_price_shows_pix_and_hides_the_expired_promotion(context: ToolContext) -> None:
    product = search_products(context, modelo="Takamine GD20")["produtos"][0]
    assert product["preco"] == "R$ 2.199,00"
    assert product["preco_no_pix"] == "R$ 2.089,05"
    assert "desconto" not in product
    assert "campanha" not in product


def test_active_promotion_shows_original_and_percent_without_pix(context: ToolContext) -> None:
    product = search_products(context, modelo="Ohana CK-20")["produtos"][0]
    assert product["preco_original"] == "R$ 549,00"
    assert product["desconto"] == "20%"
    assert product["preco_promocional"] == "R$ 439,20"
    assert product["campanha"] == "Desconto Exclusivo"
    assert "preco_no_pix" not in product
    assert product["desconto_no_pix"].startswith("não se aplica")


def test_exact_match_does_not_bring_similar_models(context: ToolContext) -> None:
    assert _ids(search_products(context, modelo="Yamaha C40")) == {81}


def test_misspelled_model_is_flagged_as_approximate(context: ToolContext) -> None:
    result = search_products(context, modelo="Tacamine GD20")
    assert 95 in _ids(result)
    assert "parecido" in result["observacao"]


def test_brand_alone_is_a_valid_model_search(context: ToolContext) -> None:
    assert len(_ids(search_products(context, modelo="Takamine"))) == 4


@pytest.mark.parametrize("instrument", ["saxofone", "trompete", "flauta", "violoncelo"])
def test_instruments_outside_the_catalog_return_nothing(
    context: ToolContext, instrument: str
) -> None:
    by_model = search_products(context, modelo=instrument)
    by_feature = search_products(context, caracteristica=instrument)
    assert by_model["total_encontrado"] == 0
    assert by_feature["total_encontrado"] == 0
    assert "sem afirmar" in by_model["observacao"]


def test_empty_category_returns_nothing(context: ToolContext) -> None:
    result = search_products(context, categoria="Cordas Orquestrais")
    assert result["total_encontrado"] == 0


def test_empty_result_offers_only_categories_that_have_products(context: ToolContext) -> None:
    result = search_products(context, categoria="Instrumentos de Sopro (Metais)")
    offered = result["categorias_com_produtos_disponiveis"]
    assert "Violões" in offered
    assert "Instrumentos de Sopro (Metais)" not in offered
    assert "Cordas Orquestrais" not in offered


def test_out_of_stock_model_is_reported_with_alternatives(context: ToolContext) -> None:
    product = search_products(context, modelo="Giannini GF-3D")["produtos"][0]
    assert product["disponibilidade"] == "temporariamente indisponível"
    assert product["preco"] == "R$ 799,90"
    alternatives = product["alternativas"]
    assert alternatives
    assert all(alt["disponibilidade"] in ("disponível", "últimas unidades") for alt in alternatives)
    assert all(alt["categoria"] == "Violões" for alt in alternatives)


def test_discontinued_with_stock_is_not_sold(context: ToolContext) -> None:
    product = search_products(context, modelo="Ibanez Bass 2X")["produtos"][0]
    assert product["disponibilidade"] == "fora de catálogo"
    assert "R$" not in product["preco"]
    assert product["alternativas"]


def test_coming_soon_is_not_sold(context: ToolContext) -> None:
    product = search_products(context, modelo="KA-ABP-CTG")["produtos"][0]
    assert product["disponibilidade"] == "em breve"
    assert "R$" not in product["preco"]


def test_stock_counts_never_reach_the_agent(context: ToolContext) -> None:
    results = [
        search_products(context, modelo="Takamine GD20"),
        search_products(context, categoria="Violões"),
        get_product_details(context, [95]),
        summarize_catalog(context),
    ]
    for result in results:
        assert "stock_quantity" not in json.dumps(result)
        assert "quantidade" not in json.dumps(result)


def test_low_stock_is_shown_as_a_label(context: ToolContext) -> None:
    # Martin D-28 has 2 units in the source data.
    product = search_products(context, modelo="Martin D-28 Dreadnought")["produtos"][0]
    assert product["disponibilidade"] == "últimas unidades"


def test_normalized_string_filter_finds_steel_strings(context: ToolContext) -> None:
    result = search_products(
        context, categoria="Violões", cordas="aço", eletrico=True, preco_max=2000
    )
    assert _ids(result) == {103, 104, 107}


def test_color_filter_unifies_black_and_preto(context: ToolContext) -> None:
    assert _ids(search_products(context, categoria="Violões", cor="preto")) == {89, 104}
    assert _ids(search_products(context, categoria="Teclados e Pianos", cor="preto")) == {
        142,
        143,
        144,
    }


def test_seven_string_filter(context: ToolContext) -> None:
    assert _ids(search_products(context, numero_de_cordas=7)) == {109, 110, 111, 112}


def test_cheapest_guitar(context: ToolContext) -> None:
    result = search_products(context, categoria="Guitarras", ordenar="menor_preco", limite=1)
    assert _ids(result) == {134}
    assert "Mostrando 1 de 5" in result["observacao"]


def test_most_expensive_of_the_store(context: ToolContext) -> None:
    result = search_products(context, ordenar="maior_preco", limite=1)
    assert _ids(result) == {143}


def test_promotions_list_only_active_ones(context: ToolContext) -> None:
    assert _ids(search_products(context, apenas_promocao=True)) == {90, 94, 121, 127}


def test_brand_filter_handles_two_word_brands(context: ToolContext) -> None:
    ids = _ids(search_products(context, marca="music man"))
    assert ids == {135, 138}


def test_characteristic_with_category_only_orders_the_category(context: ToolContext) -> None:
    result = search_products(context, categoria="Ukuleles", caracteristica="viagem", limite=30)
    assert result["total_encontrado"] == 14
    assert all(product["categoria"] == "Ukuleles" for product in result["produtos"])
    # The model once read a category total as "33 options for beginners".
    assert "não de modelos indicados" in result["sobre_o_total"]


def test_characteristic_without_category_requires_the_words(context: ToolContext) -> None:
    result = search_products(context, caracteristica="iniciante")
    assert 81 in _ids(result)


def test_details_translate_spec_labels(context: ToolContext) -> None:
    product = get_product_details(context, [81])["produtos"][0]
    assert product["especificacoes"]["tampo"] == "Spruce"
    assert "descricao" in product


def test_details_omit_descriptions_that_contradict_the_product(context: ToolContext) -> None:
    product = get_product_details(context, [142])["produtos"][0]
    assert "descricao" not in product
    assert product["especificacoes"]["teclas"] == "61"
    assert "88" not in json.dumps(product)


def test_details_report_unknown_ids(context: ToolContext) -> None:
    assert get_product_details(context, [81, 9999])["ids_nao_encontrados"] == [9999]


def test_summary_lists_empty_categories_with_zero(context: ToolContext) -> None:
    summary = {entry["categoria"]: entry for entry in summarize_catalog(context)["categorias"]}
    assert summary["Cordas Orquestrais"]["disponiveis"] == 0
    assert summary["Guitarras"]["disponiveis"] == 5
    assert summary["Guitarras"]["menor_preco"] == "R$ 5.400,00"
    assert "Music Man" in summary["Baixos"]["marcas"]


def test_summary_of_one_category(context: ToolContext) -> None:
    assert len(summarize_catalog(context, "Violões")["categorias"]) == 1


def test_tools_expose_schema_and_return_json(context: ToolContext) -> None:
    tools = {tool.name: tool for tool in build_catalog_tools(context)}
    assert set(tools) == {"buscar_produtos", "detalhar_produtos", "resumir_catalogo"}
    output = tools["buscar_produtos"].invoke({"modelo": "Yamaha C40"})
    assert json.loads(output)["produtos"][0]["id"] == 81
