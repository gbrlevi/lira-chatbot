"""Terminal chat with the agent.

Progress is shown live, but the answer only appears after the output guard approves it:
streaming tokens would show the customer text that may still be discarded.
"""

import argparse
import sys
from uuid import uuid4

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from app.llm import MissingApiKeyError
from app.service import ChatEvent, ChatService, StoreNotBuiltError, create_chat_service

EXIT_WORDS = {"sair", "exit", "quit"}


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Converse com a Lira, da Empório da Música.")
    parser.add_argument("--thread", help="Retoma uma conversa pelo identificador.")
    parser.add_argument(
        "--debug", action="store_true", help="Mostra as ferramentas usadas e seus retornos."
    )
    parser.add_argument("-m", "--message", help="Envia uma única mensagem e encerra.")
    return parser.parse_args(argv)


def _event_printer(console: Console, debug: bool):
    def show(event: ChatEvent) -> None:
        if event.kind == "tool_start":
            detail = f" {event.args}" if debug else ""
            console.print(Text(f"  consultando {event.name}{detail}", style="dim"))
        elif event.kind == "tool_end" and debug:
            console.print(Text(f"  retorno: {event.output}", style="dim"))

    return show


def _turn(service: ChatService, console: Console, thread: str, text: str, debug: bool) -> None:
    reply = service.reply(thread, text, on_event=_event_printer(console, debug))
    console.print(Panel(Text(reply.text), title="Lira", title_align="left", border_style="cyan"))


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = _parse_args(argv)
    console = Console()

    try:
        service = create_chat_service()
    except (MissingApiKeyError, StoreNotBuiltError) as error:
        console.print(f"[red]{error}[/red]")
        return 1

    thread = args.thread or uuid4().hex
    if args.message:
        _turn(service, console, thread, args.message, args.debug)
        return 0

    console.print(f"[dim]Conversa {thread}. Para retomar: --thread {thread}. Digite 'sair'.[/dim]")
    while True:
        try:
            text = console.input("[bold]Você:[/bold] ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if text.lower() in EXIT_WORDS:
            break
        _turn(service, console, thread, text, args.debug)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
