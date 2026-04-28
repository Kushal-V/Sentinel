"""Langfuse observability layer.

Attaches a Langfuse ``CallbackHandler`` to LangChain LLM instances so every
``ChatGroq`` invocation emits structured traces — token counts, latency,
prompt/completion text, cost, model name.

If ``LANGFUSE_PUBLIC_KEY`` or ``LANGFUSE_SECRET_KEY`` is not set in the
environment, this module is a zero-overhead no-op. The Sentinel app runs
identically with or without keys: ``get_callbacks()`` always returns a
list, the orchestrator passes that list to ``ChatGroq(callbacks=...)``,
and an empty list is a fully supported LangChain configuration.

Env vars consumed:
    LANGFUSE_PUBLIC_KEY   (required to enable)
    LANGFUSE_SECRET_KEY   (required to enable)
    LANGFUSE_HOST         (optional, defaults to https://cloud.langfuse.com)

Defensive contract:
    Any failure during import, instantiation, or handler creation is
    caught and logged. The app keeps running with tracing disabled. We
    never let observability break Sentinel's core supply-chain duties.
"""
from __future__ import annotations

import logging
import os
import threading
from typing import Any, Optional

_LOG = logging.getLogger(__name__)

# Module-level state. Guarded by ``_INIT_LOCK`` so concurrent Streamlit
# reruns cannot double-instantiate the handler.
_LANGFUSE_HANDLER: Optional[Any] = None
_INIT_LOCK = threading.Lock()
_INIT_DONE: bool = False


def is_enabled() -> bool:
    """Return ``True`` iff both Langfuse credentials are present in env.

    The host variable is optional — Langfuse defaults to its hosted
    cloud endpoint when ``LANGFUSE_HOST`` is unset.
    """
    return bool(os.environ.get("LANGFUSE_PUBLIC_KEY")) and bool(
        os.environ.get("LANGFUSE_SECRET_KEY")
    )


def init_tracing() -> None:
    """Idempotently initialise the Langfuse handler.

    Safe to call multiple times — the second invocation is a no-op. This
    matters under Streamlit, which re-executes the entire script on every
    user interaction. The first call creates the handler; subsequent
    calls early-return.

    No-op when env vars are missing or when the ``langfuse`` package is
    not installed. Any unexpected failure is logged at ``exception``
    level but never propagates: tracing is best-effort, never blocking.
    """
    global _LANGFUSE_HANDLER, _INIT_DONE
    with _INIT_LOCK:
        if _INIT_DONE:
            return
        _INIT_DONE = True
        if not is_enabled():
            _LOG.info("Langfuse keys not present; observability disabled")
            return
        try:
            # Langfuse v3 LangChain integration. The ``CallbackHandler``
            # picks up the public/secret/host env vars internally.
            from langfuse.langchain import CallbackHandler  # type: ignore[import-not-found]

            _LANGFUSE_HANDLER = CallbackHandler()
            _LOG.info("Langfuse tracing initialised")
        except Exception:  # noqa: BLE001
            # Catch literally anything — bad import, network probe at
            # construct time, key validation, etc. We must not crash.
            _LOG.exception(
                "Failed to initialise Langfuse — continuing without tracing"
            )
            _LANGFUSE_HANDLER = None


def get_callback_handler() -> Optional[Any]:
    """Return the cached Langfuse ``CallbackHandler``, or ``None``.

    Lazily triggers ``init_tracing()`` on first call so callers do not
    have to remember to initialise. After the first call the cached
    handler (or ``None``) is returned directly.
    """
    if not _INIT_DONE:
        init_tracing()
    return _LANGFUSE_HANDLER


def get_callbacks() -> list:
    """Return a list suitable for ``ChatGroq(callbacks=...)``.

    Returns ``[handler]`` when tracing is active, otherwise ``[]``. An
    empty list is a valid value for the ``callbacks`` kwarg on every
    LangChain LLM client, so the call site does not need to branch.
    """
    handler = get_callback_handler()
    return [handler] if handler is not None else []


def status_summary() -> str:
    """Return a one-line status string for UI display.

    Three branches, exhaustive:
        - keys absent -> "disabled (no Langfuse keys)"
        - keys present and handler ready -> "Langfuse active"
        - keys present but init failed -> "init failed"
    """
    if not is_enabled():
        return "Tracing: disabled (no Langfuse keys)"
    if _LANGFUSE_HANDLER is not None:
        return "Tracing: Langfuse active"
    return "Tracing: enabled but handler init failed"
