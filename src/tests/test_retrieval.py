from pathlib import Path

import pytest
from fakes import FakeEmbedder

from app.ingest.documents import KIND_POLICY, KIND_PRODUCT
from app.retrieval.index import (
    HybridIndex,
    StaleIndexError,
    build_match_query,
    build_strict_match_query,
)


@pytest.fixture(scope="module")
def index(store_path: Path, embedder: FakeEmbedder) -> HybridIndex:
    return HybridIndex(store_path, embedder)


def _refs(hits) -> list[str]:
    return [hit.ref for hit in hits]


def test_match_query_quotes_terms_and_applies_light_stemming() -> None:
    assert build_match_query("devolver") == '"devol"*'


def test_match_query_drops_stopwords() -> None:
    assert build_match_query("como fazer para devolver") == '"fazer"* OR "devol"*'


def test_match_query_with_only_stopwords_is_empty() -> None:
    assert build_match_query("o que é") is None


def test_strict_match_requires_every_word() -> None:
    assert build_strict_match_query("violão iniciante") == '"viol"* AND "inician"*'


@pytest.mark.parametrize(
    "hostile",
    ['"', 'pix" OR "a', "pix AND NOT boleto", "NEAR(pix boleto)", "content:pix", "*", "(((", "'--"],
)
def test_fts_syntax_in_customer_text_cannot_break_the_search(
    index: HybridIndex, hostile: str
) -> None:
    index.search(hostile, KIND_POLICY)
    index.matching_refs(hostile, KIND_PRODUCT)


def test_exact_term_is_found_lexically(index: HybridIndex) -> None:
    assert index.search("Jadlog", KIND_POLICY)[0].ref == "5.2"


def test_stemming_connects_word_variations(index: HybridIndex) -> None:
    # The query says "devolver"; the manual says "devolução".
    assert "4.1" in _refs(index.search("quero devolver", KIND_POLICY))


def test_kind_keeps_policies_and_products_apart(index: HybridIndex) -> None:
    assert all("." in ref or int(ref) <= 10 for ref in _refs(index.search("PIX", KIND_POLICY)))
    assert all(int(ref) >= 81 for ref in _refs(index.search("Yamaha", KIND_PRODUCT)))


def test_results_respect_k(index: HybridIndex) -> None:
    assert len(index.search("loja", KIND_POLICY, k=2)) == 2


@pytest.mark.parametrize("instrument", ["saxofone", "trompete", "flauta", "violoncelo"])
def test_instruments_outside_the_catalog_match_nothing(index: HybridIndex, instrument: str) -> None:
    assert index.matching_refs(instrument, KIND_PRODUCT) == set()


def test_strict_match_accepts_plural_forms(index: HybridIndex) -> None:
    # The query says "iniciante"; descriptions say "iniciantes".
    assert "81" in index.matching_refs("iniciante", KIND_PRODUCT)


def test_ranking_covers_all_allowed_documents(index: HybridIndex) -> None:
    allowed = {str(product_id) for product_id in range(116, 131)}
    hits = index.search("presente", KIND_PRODUCT, k=30, allowed_refs=allowed)
    assert {hit.ref for hit in hits} == allowed


def test_allowed_refs_applies_hard_filters(index: HybridIndex) -> None:
    hits = index.search("Yamaha", KIND_PRODUCT, k=10, allowed_refs={"81", "82"})
    assert set(_refs(hits)) == {"81", "82"}


def test_unreliable_description_is_not_indexed(index: HybridIndex) -> None:
    # The Korg description says "Roland" and "88 teclas", which contradict the product.
    assert "142" not in index.matching_refs("88 teclas", KIND_PRODUCT)


def test_index_built_with_another_embedder_is_rejected(store_path: Path) -> None:
    class OtherEmbedder(FakeEmbedder):
        name = "fake:other"

    with pytest.raises(StaleIndexError):
        HybridIndex(store_path, OtherEmbedder())
