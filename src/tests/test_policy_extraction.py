"""Fidelity tests: if a new manual or parser breaks a critical fact, these fail."""

import pytest

from app.config import REPO_ROOT
from app.ingest.policies import PolicyChunk, chunk_markdown, extract_markdown

PDF_PATH = REPO_ROOT / "data" / "políticas_da_loja.pdf"


@pytest.fixture(scope="module")
def markdown() -> str:
    return extract_markdown(PDF_PATH)


@pytest.fixture(scope="module")
def chunks(markdown: str) -> dict[str, PolicyChunk]:
    return {chunk.section: chunk for chunk in chunk_markdown(markdown)}


@pytest.mark.parametrize(
    "row",
    [
        "| Segunda a Sexta-feira | 09:00 às 18:00 |",
        "| Sábado | 09:00 às 13:00 |",
        "| Domingo e Feriados | Fechado |",
        "| PIX | Pagamento à vista | 5% de desconto sobre o preço de tabela |",
        "| Boleto Bancário | Pagamento à vista | Compensação em até 3 dias úteis |",
        "| PAC (Correios) | 5 a 12 dias úteis | Sim | Incluído |",
        "| SEDEX (Correios) | 2 a 5 dias úteis | Sim | Incluído |",
        "| Jadlog (.package) | 3 a 8 dias úteis | Sim | Incluído |",
        "| Endereço | Rua 14 de Maio, 3200 — Centro, Campo Grande - MS, 79202-333 |",
    ],
)
def test_tables_keep_each_row_together(markdown: str, row: str) -> None:
    assert row in markdown


@pytest.mark.parametrize(
    ("section", "fact"),
    [
        ("3.1", "Parcelamento de 4x a 6x: sem juros, parcela mínima de R$ 80,00."),
        ("3.1", "sem valor mínimo de parcela (exceto abaixo de R$ 50,00)"),
        ("3.1", "para compras acima de R$ 2.000,00"),
        ("4.1", "até 7 (sete) dias corridos após o recebimento do produto"),
        ("4.1", "em até 10 dias úteis"),
        ("4.1", "O frete de devolução é por conta da loja"),
        ("4.2", "em até 30 (trinta) dias corridos após a compra"),
        ("4.4", "Boquilhas de instrumentos de sopro"),
        ("5.1", "Frete grátis para pedidos acima de R$ 500,00."),
        ("5.1", "taxa fixa de R$ 35,00"),
        ("6.2", "O desconto de PIX (5%) não se aplica sobre preços já promocionais."),
        ("7", "(67) 3321-4500"),
        ("7.3", "Prazo de retorno: 24 horas úteis."),
        ("8.1", "garantia legal de 90 (noventa) dias"),
        ("1", "não comercializamos acessórios como cordas, palhetas, cabos, cases, pedais"),
    ],
)
def test_critical_facts_land_in_the_right_section(
    chunks: dict[str, PolicyChunk], section: str, fact: str
) -> None:
    assert fact in chunks[section].text


def test_every_numbered_section_becomes_a_chunk(chunks: dict[str, PolicyChunk]) -> None:
    expected = {
        "1", "1.1", "1.2", "2", "3", "3.1", "4", "4.1", "4.2", "4.3", "4.4", "5.1", "5.2",
        "5.3", "6.1", "6.2", "7", "7.1", "7.2", "7.3", "8.1", "8.2", "8.3", "9", "10",
    }  # fmt: skip
    assert set(chunks) == expected


def test_page_furniture_is_removed(markdown: str) -> None:
    assert "Manual de Políticas e Procedimentos" not in markdown
    assert "Página 3" not in markdown
    assert "Última atualização" not in markdown


def test_subsection_title_keeps_its_parent_for_context(chunks: dict[str, PolicyChunk]) -> None:
    assert chunks["4.1"].title == (
        "Política de Trocas e Devoluções: Direito de Arrependimento (Compras Online)"
    )


def test_searchable_text_cites_the_section(chunks: dict[str, PolicyChunk]) -> None:
    assert chunks["2"].searchable_text.startswith("Seção 2 — Horário de Funcionamento\n")


def test_chunks_are_small_enough_to_cite_precisely(chunks: dict[str, PolicyChunk]) -> None:
    assert max(len(chunk.text) for chunk in chunks.values()) < 1000
