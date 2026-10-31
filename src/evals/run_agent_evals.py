"""Runs the eval cases against the real model.

    uv run python src/evals/run_agent_evals.py --repeat 2
    uv run python src/evals/run_agent_evals.py --model gemini-3.8-flash

Results are saved after each case, so an interrupted run continues where it stopped. The
pause between calls keeps the run inside the free tier limits.
"""

import argparse
import json
import logging
import os
import statistics
import sys
import tempfile
import time
from pathlib import Path

from cases import CASES, Case
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from app.agent.graph import RECURSION_LIMIT, build_graph
from app.config import get_settings
from app.llm import build_chat_model
from app.retrieval.embeddings import get_embedder
from app.retrieval.index import HybridIndex
from app.runtime_store import RuntimeStore
from app.text import strip_accents
from app.tools.context import ToolContext
from app.tools.registry import build_tools

RESULTS_DIR = Path(__file__).parent / "results"
PAUSE_BETWEEN_TURNS_SECONDS = 4.0


def _normalized(text: str) -> str:
    return strip_accents(text).lower()


def _evaluate(case: Case, answer: str, tools_used: set[str]) -> list[str]:
    """Return the failed checks. An empty list means the case passed."""
    failures = []
    text = _normalized(answer)
    for tool in sorted(case.required_tools - tools_used):
        failures.append(f"não usou {tool}")
    for tool in sorted(case.forbidden_tools & tools_used):
        failures.append(f"usou {tool} indevidamente")
    for expected in case.must_contain:
        if not any(_normalized(option) in text for option in expected.split("|")):
            failures.append(f"faltou: {expected}")
    for forbidden in case.must_not_contain:
        if _normalized(forbidden) in text:
            failures.append(f"citou: {forbidden}")
    return failures


def _set_moment(moment: str) -> None:
    os.environ["REFERENCE_DATETIME"] = moment
    get_settings.cache_clear()


def _run_case(case: Case, run_number: int, model: str, tools: list) -> dict:
    _set_moment(case.moment)
    graph = build_graph(build_chat_model(model=model), tools, checkpointer=InMemorySaver())
    config = {
        "configurable": {"thread_id": f"{case.id}-{run_number}"},
        "recursion_limit": RECURSION_LIMIT,
    }
    started = time.perf_counter()
    error = None
    messages: list = []
    for turn in case.turns:
        try:
            messages = graph.invoke({"messages": [HumanMessage(turn)]}, config=config)["messages"]
        except Exception as exception:  # noqa: BLE001 - a failed case is data for the report
            error = f"{type(exception).__name__}: {str(exception)[:200]}"
            break
        time.sleep(PAUSE_BETWEEN_TURNS_SECONDS)
    elapsed = time.perf_counter() - started - PAUSE_BETWEEN_TURNS_SECONDS * len(case.turns)

    calls = [call for m in messages if isinstance(m, AIMessage) for call in m.tool_calls]
    tools_used = {call["name"] for call in calls}
    usages = [m.usage_metadata for m in messages if isinstance(m, AIMessage) and m.usage_metadata]
    answer = messages[-1].text if messages and isinstance(messages[-1], AIMessage) else ""
    failures = [error] if error else _evaluate(case, answer, tools_used)
    return {
        "case": case.id,
        "category": case.category,
        "run": run_number,
        "passed": not failures,
        "provider_error": bool(error),
        "failures": failures,
        "answer": answer,
        "tool_calls": [{"name": call["name"], "args": call["args"]} for call in calls],
        "seconds": round(max(elapsed, 0.0), 2),
        "input_tokens": sum(usage.get("input_tokens", 0) for usage in usages),
        "output_tokens": sum(usage.get("output_tokens", 0) for usage in usages),
    }


def _load_previous(path: Path) -> dict[tuple[str, int], dict]:
    if not path.exists():
        return {}
    results = json.loads(path.read_text(encoding="utf-8"))["results"]
    # Provider errors say nothing about the agent, so those cases run again.
    return {(r["case"], r["run"]): r for r in results if not r["provider_error"]}


def _summary(results: list[dict]) -> dict:
    by_category: dict[str, dict[str, int]] = {}
    for result in results:
        entry = by_category.setdefault(result["category"], {"passed": 0, "total": 0})
        entry["total"] += 1
        entry["passed"] += result["passed"]
    answered = [result for result in results if not result["provider_error"]]
    return {
        "passed": sum(result["passed"] for result in results),
        "total": len(results),
        "provider_errors": len(results) - len(answered),
        "by_category": by_category,
        "median_seconds_per_case": (
            statistics.median(result["seconds"] for result in answered) if answered else None
        ),
        "average_input_tokens": (
            round(statistics.mean(result["input_tokens"] for result in answered))
            if answered
            else None
        ),
        "average_output_tokens": (
            round(statistics.mean(result["output_tokens"] for result in answered))
            if answered
            else None
        ),
    }


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    logging.getLogger("langchain_google_genai").setLevel(logging.ERROR)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=get_settings().llm_model)
    parser.add_argument("--repeat", type=int, default=1, help="Quantas vezes rodar cada caso.")
    parser.add_argument("--case", help="Roda só os casos cujo id contém este texto.")
    parser.add_argument("--fresh", action="store_true", help="Ignora resultados anteriores.")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(exist_ok=True)
    output = RESULTS_DIR / f"{args.model}.json"
    selected = [case for case in CASES if not args.case or args.case in case.id]
    known = {} if args.fresh or args.case else _load_previous(output)

    settings = get_settings()
    with tempfile.TemporaryDirectory() as directory:
        runtime = RuntimeStore(Path(directory) / "runtime.db")
        context = ToolContext(
            settings.store_db_path,
            HybridIndex(settings.store_db_path, get_embedder()),
            runtime,
        )
        tools = build_tools(context)

        results = []
        for run in range(1, args.repeat + 1):
            for case in selected:
                result = known.get((case.id, run)) or _run_case(case, run, args.model, tools)
                results.append(result)
                mark = "ok   " if result["passed"] else "FALHA"
                print(f"{mark} [{run}] {case.id} {result['failures'] or ''}", flush=True)
                if not args.case:
                    summary = _summary(results)
                    report = {"model": args.model, "summary": summary, "results": results}
                    output.write_text(
                        json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8"
                    )

    summary = _summary(results)
    print(f"\n{args.model}: {summary['passed']}/{summary['total']} aprovados")
    print(f"Erros de provedor: {summary['provider_errors']}")
    for category, entry in summary["by_category"].items():
        print(f"  {category}: {entry['passed']}/{entry['total']}")
    print(f"Mediana por caso: {summary['median_seconds_per_case']} s")
    print(
        f"Tokens médios por caso: {summary['average_input_tokens']} entrada, "
        f"{summary['average_output_tokens']} saída"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
