from pathlib import Path

import numpy as np
import pytest
from fakes import FakeEmbedder

from app.ingest.documents import KIND_POLICY, KIND_PRODUCT
from app.retrieval.index import HybridIndex, StaleIndexError, build_match_query


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


@pytest.mark.parametrize(
    "hostile",
    ['"', 'pix" OR "a', "pix AND NOT boleto", "NEAR(pix boleto)", "content:pix", "*", "(((", "'--"],
)
def test_fts_syntax_in_customer_text_cannot_break_the_search(
    index: HybridIndex, hostile: str
) -> None:
    index.search(hostile, KIND_POLICY)


def test_exact_term_is_found_lexically(index: HybridIndex) -> None:
    hits = index.search("Jadlog", KIND_POLICY)
    assert hits[0].ref == "5.2"
    assert hits[0].lexical_match


def test_stemming_connects_word_variations(index: HybridIndex) -> None:
    # The query says "devolver"; the manual says "devolução".
    assert "4.1" in _refs(index.search("quero devolver", KIND_POLICY))


def test_kind_keeps_policies_and_products_apart(index: HybridIndex) -> None:
    assert all("." in ref or int(ref) <= 10 for ref in _refs(index.search("PIX", KIND_POLICY)))
    assert all(int(ref) >= 81 for ref in _refs(index.search("Yamaha", KIND_PRODUCT)))


def test_results_respect_k(index: HybridIndex) -> None:
    assert len(index.search("loja", KIND_POLICY, k=2)) == 2


def test_min_similarity_lets_irrelevant_queries_return_nothing(index: HybridIndex) -> None:
    assert index.search("saxofone", KIND_PRODUCT, min_similarity=0.5) == []


def test_lexical_match_qualifies_even_below_min_similarity(index: HybridIndex) -> None:
    hits = index.search("Takamine", KIND_PRODUCT, min_similarity=0.99)
    assert hits and all(hit.lexical_match for hit in hits)


def test_allowed_refs_applies_hard_filters(index: HybridIndex) -> None:
    hits = index.search("Yamaha", KIND_PRODUCT, k=10, allowed_refs={"81", "82"})
    assert set(_refs(hits)) == {"81", "82"}


def test_unreliable_description_is_not_indexed(index: HybridIndex) -> None:
    # The Korg description says "Roland" and "88 teclas", which contradict the product.
    hits = index.search("88 teclas", KIND_PRODUCT, k=65, allowed_refs={"142"})
    assert hits == [] or "88" not in hits[0].text


def test_index_built_with_another_embedder_is_rejected(store_path: Path) -> None:
    class OtherEmbedder(FakeEmbedder):
        name = "fake:other"

    with pytest.raises(StaleIndexError):
        HybridIndex(store_path, OtherEmbedder())


def test_vectors_are_compared_by_cosine(index: HybridIndex) -> None:
    hit = index.search("Yamaha C40 Nylon Natural", KIND_PRODUCT)[0]
    assert 0 < hit.similarity <= 1 + np.finfo(np.float32).eps
