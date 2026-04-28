"""Sentinel observability package.

Provides a thin, defensive wrapper around the Langfuse LangChain
``CallbackHandler``. The handler is auto-attached to every ``ChatGroq``
client constructed by the orchestrator and the inter-agent tool client
when the appropriate environment variables are set.

Public surface:
    - ``init_tracing``   : idempotent initialiser, safe to call repeatedly
    - ``get_callbacks``  : returns ``[handler]`` or ``[]`` (always a list)
    - ``status_summary`` : one-line UI status string
    - ``is_enabled``     : whether the env-var contract is satisfied
"""
from src.observability.tracing import (
    get_callback_handler,
    get_callbacks,
    init_tracing,
    is_enabled,
    status_summary,
)

__all__ = [
    "get_callback_handler",
    "get_callbacks",
    "init_tracing",
    "is_enabled",
    "status_summary",
]
