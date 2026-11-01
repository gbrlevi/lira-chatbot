"""Compares retrieval strategies and embedding models on the policy manual.

    uv run python src/evals/retrieval_benchmark.py

Each question lists the sections that answer it. A question counts as found when any of them
appears in the top results. The gold set was written by hand from the manual.
"""

import json
import statistics
import sys
import tempfile
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np

from app.agent.resilience import is_transient
from app.config import get_settings
from app.db import connect_readonly
from app.ingest.documents import KIND_POLICY
from app.ingest.run import build_store
from app.retrieval.embeddings import Embedder, FastEmbedEmbedder, GeminiEmbedder
from app.retrieval.index import HybridIndex, build_match_query

K = 4
RESULT_PATH = Path(__file__).parent / "results" / "retrieval_benchmark.json"
# Pause between Gemini calls, to stay inside the free tier limits.
GEMINI_PAUSE_SECONDS = 1.5
RATE_LIMIT_ATTEMPTS = 6
RATE_LIMIT_WAIT_SECONDS = 65

# (question, sections that answer it)
GOLD: tuple[tuple[str, frozenset[str]], ...] = tuple(
    (question, frozenset(sections))
    for question, sections in (
        ("Qual o endereço da loja?", {"1.2"}),
        ("Onde vocês ficam?", {"1.2"}),
        ("Qual o telefone de vocês?", {"1.2", "7"}),
        ("Até que horas abre no sábado?", {"2"}),
        ("Vocês abrem domingo?", {"2"}),
        ("Qual o horário de funcionamento?", {"2"}),
        ("Aceita PIX?", {"3"}),
        ("Tem desconto para pagamento à vista?", {"3", "6.2"}),
        ("Quanto tempo o boleto demora para compensar?", {"3"}),
        ("Posso parcelar em 12 vezes?", {"3", "3.1"}),
        ("Qual a parcela mínima no cartão?", {"3", "3.1"}),
        ("Dá para pagar metade no pix e metade no cartão?", {"3.1"}),
        ("Me arrependi da compra, posso devolver?", {"4.1"}),
        ("Comprei pela internet e não gostei, e agora?", {"4.1"}),
        ("Quem paga o frete da devolução?", {"4.1"}),
        ("Em quanto tempo recebo meu dinheiro de volta?", {"4.1"}),
        ("Meu violão veio com defeito de fábrica", {"4.2"}),
        ("O instrumento quebrou depois de um mês, o que faço?", {"4.2", "8.1", "8.2"}),
        ("Quero trocar a cor do meu ukulele", {"4.3"}),
        ("Posso trocar um produto da liquidação?", {"4.4"}),
        ("Posso devolver a boquilha do meu saxofone?", {"4.4"}),
        ("Quanto custa o frete em Campo Grande?", {"5.1"}),
        ("Tem frete grátis?", {"5.1"}),
        ("Em quantos dias chega em Campo Grande?", {"5.1"}),
        ("Vocês entregam em Dourados?", {"5.2"}),
        ("Quanto tempo demora o SEDEX?", {"5.2"}),
        ("O produto chegou amassado na entrega", {"5.2"}),
        ("Como rastreio meu pedido?", {"5.3"}),
        ("Quando vai ser a Black Friday?", {"6.1"}),
        ("O desconto do pix vale em promoção?", {"6.2"}),
        ("Se o produto esgotar na promoção eu tenho direito ao preço?", {"6.2"}),
        ("Qual o número do WhatsApp?", {"7", "1.2"}),
        ("Quero fazer uma reclamação", {"7.3"}),
        ("A promoção que eu vi acabou, ainda vale?", {"7.3"}),
        ("Qual o prazo da garantia?", {"8.1", "8.2"}),
        ("A garantia cobre corda arrebentada?", {"8.3"}),
        ("Quero que apaguem meus dados", {"9"}),
        ("O que vocês fazem com meus dados pessoais?", {"9"}),
        ("Vocês vendem palhetas?", {"1"}),
        ("Vende amplificador?", {"1", "7.1"}),
    )
)

Ranker = Callable[[str], list[str]]


class PatientEmbedder:
    """Waits out free tier rate limits, so a benchmark run does not die halfway."""

    def __init__(self, embedder: Embedder) -> None:
        self._embedder = embedder
        self.name = embedder.name

    def _patiently(self, call: Callable, argument) -> np.ndarray:
        for _ in range(RATE_LIMIT_ATTEMPTS):
            try:
                return call(argument)
            except Exception as error:  # noqa: BLE001 - only transient errors are retried below
                if not is_transient(error):
                    raise
                print(f"    limite do provedor, aguardando {RATE_LIMIT_WAIT_SECONDS}s", flush=True)
                time.sleep(RATE_LIMIT_WAIT_SECONDS)
        return call(argument)

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return self._patiently(self._embedder.embed_documents, texts)

    def embed_query(self, text: str) -> np.ndarray:
        return self._patiently(self._embedder.embed_query, text)


def _lexical_ranker(store_path: Path) -> Ranker:
    def rank(question: str) -> list[str]:
        expression = build_match_query(question)
        if expression is None:
            return []
        with connect_readonly(store_path) as connection:
            rows = connection.execute(
                """
                SELECT d.ref FROM documents_fts f JOIN documents d ON d.doc_id = f.rowid
                WHERE documents_fts MATCH ? AND f.kind = ? ORDER BY bm25(documents_fts)
                """,
                (expression, KIND_POLICY),
            ).fetchall()
        return [row["ref"] for row in rows]

    return rank


def _dense_ranker(store_path: Path, embedder: Embedder) -> Ranker:
    with connect_readonly(store_path) as connection:
        rows = connection.execute(
            "SELECT ref, vector FROM documents WHERE kind = ?", (KIND_POLICY,)
        ).fetchall()
    refs = [row["ref"] for row in rows]
    matrix = np.stack([np.frombuffer(row["vector"], dtype=np.float32) for row in rows])

    def rank(question: str) -> list[str]:
        similarities = matrix @ embedder.embed_query(question)
        return [refs[int(position)] for position in np.argsort(-similarities)]

    return rank


def _hybrid_ranker(store_path: Path, embedder: Embedder) -> Ranker:
    index = HybridIndex(store_path, embedder)

    def rank(question: str) -> list[str]:
        return [hit.ref for hit in index.search(question, KIND_POLICY, k=K)]

    return rank


def _measure(name: str, ranker: Ranker, pause: float = 0.0) -> dict:
    found_at: list[int | None] = []
    latencies: list[float] = []
    misses: list[str] = []
    for question, expected in GOLD:
        started = time.perf_counter()
        ranking = ranker(question)
        latencies.append(time.perf_counter() - started)
        position = next((i for i, ref in enumerate(ranking[:K], 1) if ref in expected), None)
        found_at.append(position)
        if position is None:
            misses.append(question)
        time.sleep(pause)
    result = {
        "strategy": name,
        "questions": len(GOLD),
        f"recall_at_{K}": round(sum(p is not None for p in found_at) / len(GOLD), 3),
        "recall_at_1": round(sum(p == 1 for p in found_at) / len(GOLD), 3),
        "mrr": round(statistics.mean(1 / p if p else 0.0 for p in found_at), 3),
        "median_ms": round(statistics.median(latencies) * 1000, 1),
        "misses": misses,
    }
    print(
        f"{name:32} recall@{K}={result[f'recall_at_{K}']:.3f} recall@1={result['recall_at_1']:.3f}"
        f" mrr={result['mrr']:.3f} mediana={result['median_ms']}ms"
    )
    for question in misses:
        print(f"    não achou: {question}")
    return result


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    settings = get_settings()
    results = []

    with tempfile.TemporaryDirectory() as directory:
        fast_store = Path(directory) / "fast.db"
        fast = FastEmbedEmbedder()
        build_store(settings.data_dir, fast_store, fast)
        results.append(_measure("bm25", _lexical_ranker(fast_store)))
        results.append(_measure("vetorial fastembed", _dense_ranker(fast_store, fast)))
        results.append(_measure("hibrido fastembed", _hybrid_ranker(fast_store, fast)))

        if settings.ai_api_key:
            gemini_store = Path(directory) / "gemini.db"
            gemini = PatientEmbedder(GeminiEmbedder())
            build_store(settings.data_dir, gemini_store, gemini)
            results.append(
                _measure(
                    "vetorial gemini",
                    _dense_ranker(gemini_store, gemini),
                    pause=GEMINI_PAUSE_SECONDS,
                )
            )
            results.append(
                _measure(
                    "hibrido gemini",
                    _hybrid_ranker(gemini_store, gemini),
                    pause=GEMINI_PAUSE_SECONDS,
                )
            )
        else:
            print("AI_API_KEY ausente: Gemini Embedding não foi avaliado.")

    RESULT_PATH.parent.mkdir(exist_ok=True)
    RESULT_PATH.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
