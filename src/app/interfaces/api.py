"""HTTP access to the agent, the entry point a messaging channel would call.

The handlers are plain functions on purpose: FastAPI runs them in a thread pool, which suits
the synchronous graph. Authentication is left out of this PoC and listed in the README.
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated
from uuid import uuid4

import uvicorn
from fastapi import Depends, FastAPI, Request
from pydantic import BaseModel, Field

from app.service import ChatService, create_chat_service

# Larger messages are rejected here; the agent's own, lower limit answers politely.
MAX_REQUEST_CHARS = 10_000


class ChatRequest(BaseModel):
    message: str = Field(max_length=MAX_REQUEST_CHARS, description="Mensagem do cliente.")
    thread_id: str | None = Field(
        default=None, max_length=200, description="Conversa a continuar. Omita para iniciar outra."
    )


class ToolUseResponse(BaseModel):
    name: str
    args: dict


class ChatResponse(BaseModel):
    thread_id: str
    reply: str
    tools_used: list[ToolUseResponse]
    degraded: bool = Field(description="Verdadeiro quando o provedor falhou e a resposta é fixa.")


@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncIterator[None]:
    if not hasattr(application.state, "chat_service"):
        application.state.chat_service = create_chat_service()
    yield


app = FastAPI(title="Lira — Empório da Música", version="0.1.0", lifespan=lifespan)


def get_chat_service(request: Request) -> ChatService:
    return request.app.state.chat_service


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/chat")
def chat(
    payload: ChatRequest, service: Annotated[ChatService, Depends(get_chat_service)]
) -> ChatResponse:
    thread_id = payload.thread_id or uuid4().hex
    reply = service.reply(thread_id, payload.message)
    return ChatResponse(
        thread_id=thread_id,
        reply=reply.text,
        tools_used=[ToolUseResponse(name=use.name, args=use.args) for use in reply.tools_used],
        degraded=reply.degraded,
    )


def main() -> None:
    uvicorn.run("app.interfaces.api:app", host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()
