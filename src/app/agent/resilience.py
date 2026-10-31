"""Tells provider failures worth retrying from those that would just fail again."""

import httpx
from google.genai import errors as genai_errors

_TRANSIENT_STATUS_CODES = frozenset({408, 429, 500, 502, 503, 504})
_MAX_CAUSE_DEPTH = 10


def is_transient(error: BaseException | None) -> bool:
    """Walk the exception chain, because the LangChain client wraps the provider's error."""
    for _ in range(_MAX_CAUSE_DEPTH):
        if error is None:
            return False
        if isinstance(error, genai_errors.APIError):
            return error.code in _TRANSIENT_STATUS_CODES
        if isinstance(error, httpx.TransportError | TimeoutError | ConnectionError):
            return True
        error = error.__cause__ or error.__context__
    return False
