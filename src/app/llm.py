"""Single place that knows how to build the chat model."""

from langchain.chat_models import init_chat_model
from langchain_core.language_models.chat_models import BaseChatModel

from app.config import get_settings


class MissingApiKeyError(RuntimeError):
    """Raised when a model is requested without AI_API_KEY configured."""


def build_chat_model(*, model: str | None = None) -> BaseChatModel:
    settings = get_settings()
    if not settings.ai_api_key:
        raise MissingApiKeyError("Defina AI_API_KEY no arquivo .env (veja .env.example).")

    # Temperature is not set because the default model uses fixed sampling and ignores it.
    # Client retries stay off because the graph owns the retry policy. Retrying at both
    # levels would multiply the wait whenever the provider is unavailable.
    return init_chat_model(
        model or settings.llm_model,
        model_provider="google_genai",
        api_key=settings.ai_api_key,
        timeout=settings.llm_timeout_seconds,
        max_retries=1,
    )
