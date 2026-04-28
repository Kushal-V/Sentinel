"""tests/test_tracing.py

Tests for ``src/observability/tracing.py``.

These tests do NOT require any Langfuse account or network access. They
verify the defensive contract: the module is a zero-overhead no-op when
credentials are absent, the API surface always returns sane values, and
any failure during init is logged but never propagates.
"""
from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def reset_tracing_module(monkeypatch):
    """Reset module-level state between tests.

    The tracing module memoises whether init has run. Clear that and the
    cached handler so each test sees a fresh slate. We also clear the
    Langfuse env vars so the default branch is "disabled" — individual
    tests that need keys re-set them explicitly.
    """
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_HOST", raising=False)

    import src.observability.tracing as t
    t._LANGFUSE_HANDLER = None
    t._INIT_DONE = False
    yield
    # Reset after the test as well so order-dependence cannot leak.
    t._LANGFUSE_HANDLER = None
    t._INIT_DONE = False


def test_disabled_when_keys_missing():
    """No env vars -> ``is_enabled`` False, no handler, empty callbacks."""
    from src.observability import tracing

    assert tracing.is_enabled() is False
    tracing.init_tracing()
    assert tracing.get_callback_handler() is None
    assert tracing.get_callbacks() == []
    assert "disabled" in tracing.status_summary().lower()


def test_init_is_idempotent():
    """Calling init repeatedly must not crash or double-instantiate."""
    from src.observability import tracing

    tracing.init_tracing()
    tracing.init_tracing()  # second call must be a no-op
    tracing.init_tracing()  # third call too
    # Still returns the same (None) handler in disabled mode.
    assert tracing.get_callback_handler() is None


def test_get_callbacks_returns_empty_list_when_disabled():
    """Contract: ``get_callbacks`` always returns a list."""
    from src.observability import tracing

    cbs = tracing.get_callbacks()
    assert cbs == []
    assert isinstance(cbs, list)


def test_get_callback_handler_lazily_initialises():
    """Calling ``get_callback_handler`` before ``init_tracing`` must work."""
    import src.observability.tracing as t
    # Confirm reset fixture left _INIT_DONE False.
    assert t._INIT_DONE is False
    handler = t.get_callback_handler()
    assert handler is None
    assert t._INIT_DONE is True  # lazy init flipped the flag


def test_enabled_with_keys_does_not_crash(monkeypatch):
    """With keys set, init must complete without raising.

    Either the langfuse package is installed and the handler is built
    (active), or the package is missing / rejects fake keys (init
    failed). Both branches are acceptable; the contract is "no crash"
    and a sensible status string.
    """
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test-fake")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test-fake")
    from src.observability import tracing

    assert tracing.is_enabled() is True
    tracing.init_tracing()  # must not raise
    summary = tracing.status_summary()
    assert summary.startswith("Tracing:")
    # The list contract holds whether or not the handler initialised.
    cbs = tracing.get_callbacks()
    assert isinstance(cbs, list)


def test_failed_init_does_not_crash(monkeypatch):
    """If the langfuse import fails, the app must keep running.

    We force the import of ``langfuse.langchain`` to raise, then call
    ``init_tracing`` and assert the module degrades gracefully.
    """
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")

    import builtins
    import sys

    # Drop any previously-loaded langfuse modules so our patched import
    # is the path that runs.
    for mod in list(sys.modules):
        if mod.startswith("langfuse"):
            sys.modules.pop(mod, None)

    real_import = builtins.__import__

    def _fake_import(name, *args, **kwargs):
        if name == "langfuse.langchain" or name.startswith("langfuse"):
            raise ImportError("simulated langfuse import failure")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fake_import)

    from src.observability import tracing

    # Must not raise even though the import explodes.
    tracing.init_tracing()
    assert tracing.get_callback_handler() is None
    # Status should explicitly indicate failure, not "active".
    assert "active" not in tracing.status_summary().lower()


def test_callbacks_kwarg_accepted_by_chatgroq_signature():
    """``ChatGroq`` accepts a ``callbacks`` kwarg.

    The orchestrator wires ``callbacks=get_callbacks()`` on every client.
    LangChain's BaseChatModel base class declares ``callbacks`` as a
    valid constructor parameter. A regression in the dependency would
    silently break tracing — this guard catches that.
    """
    from langchain_groq import ChatGroq

    # Inspect the model fields. ``callbacks`` is inherited from the
    # LangChain base class.
    fields = getattr(ChatGroq, "model_fields", None)
    if fields is not None:
        assert "callbacks" in fields
    else:
        # Pydantic v1 / very old versions: fall back to attribute check.
        assert hasattr(ChatGroq, "callbacks") or "callbacks" in ChatGroq.__init__.__code__.co_varnames
