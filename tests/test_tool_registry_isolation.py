"""F6 — verify ContextVar-based session isolation."""
from __future__ import annotations

import asyncio
import sys
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.tools import tool_registry as tr


def _make_fake_manager(name: str):
    m = MagicMock()
    m.name = name  # arbitrary identifier for assertions
    return m


def test_set_and_get_in_same_context():
    a = _make_fake_manager("A")
    token = tr.set_data_manager(a)
    try:
        assert tr.get_data_manager() is a
    finally:
        tr.reset_data_manager(token)


def test_reset_restores_previous_value():
    a = _make_fake_manager("A")
    b = _make_fake_manager("B")
    t1 = tr.set_data_manager(a)
    t2 = tr.set_data_manager(b)
    assert tr.get_data_manager() is b
    tr.reset_data_manager(t2)
    assert tr.get_data_manager() is a
    tr.reset_data_manager(t1)


def test_threads_have_isolated_contexts():
    """Two threads using copy_context each see their own manager."""
    import contextvars

    a = _make_fake_manager("A")
    b = _make_fake_manager("B")
    seen = {}

    def worker(name, manager):
        token = tr.set_data_manager(manager)
        time.sleep(0.05)
        seen[name] = tr.get_data_manager()
        tr.reset_data_manager(token)

    # Run each worker in its own copied context
    ctx_a = contextvars.copy_context()
    ctx_b = contextvars.copy_context()

    t1 = threading.Thread(target=ctx_a.run, args=(worker, "A", a))
    t2 = threading.Thread(target=ctx_b.run, args=(worker, "B", b))
    t1.start(); t2.start()
    t1.join(); t2.join()

    assert seen["A"] is a
    assert seen["B"] is b


def test_default_is_none_in_fresh_context():
    """ContextVar's declared default is None.

    A pristine, separately-spawned thread (which does NOT inherit the
    pytest main thread's context) sees the ContextVar's compile-time
    default of ``None``.  ``copy_context()`` would inherit the current
    value, so we use a bare ``threading.Thread`` here to verify the
    fallback path.
    """
    seen = {}

    def check():
        # Bare thread — no inherited context. The ContextVar must
        # report its declared default of None.
        seen["dm"] = tr._data_manager_var.get()

    t = threading.Thread(target=check)
    t.start()
    t.join()
    assert seen["dm"] is None


def test_async_task_isolation():
    """Two asyncio tasks see independent managers."""
    a = _make_fake_manager("A")
    b = _make_fake_manager("B")
    seen = {}

    async def task(name, manager):
        token = tr.set_data_manager(manager)
        await asyncio.sleep(0.01)
        seen[name] = tr.get_data_manager()
        tr.reset_data_manager(token)

    async def main():
        await asyncio.gather(
            asyncio.create_task(task("A", a)),
            asyncio.create_task(task("B", b)),
        )

    asyncio.run(main())
    assert seen["A"] is a
    assert seen["B"] is b


def test_existing_tool_callers_still_work():
    """get_data_manager() returns whatever set_data_manager() most recently set in the same context."""
    m = _make_fake_manager("compat")
    token = tr.set_data_manager(m)
    try:
        # simulate tool reading via the public accessor
        assert tr.get_data_manager() is m
    finally:
        tr.reset_data_manager(token)


def test_concurrent_streamlit_simulation():
    """Simulate two simultaneous Streamlit sessions doing tool calls.

    Each thread sets its manager, then 'tools' read the manager. The
    threads use copy_context() to mimic Streamlit's per-session script
    runner thread isolation.
    """
    import contextvars
    sessions = {}

    def session_run(session_id, manager_name):
        manager = _make_fake_manager(manager_name)
        tr.set_data_manager(manager)
        # simulate several tool reads
        observed = []
        for _ in range(5):
            time.sleep(0.005)
            observed.append(tr.get_data_manager().name)
        sessions[session_id] = observed

    ctx1 = contextvars.copy_context()
    ctx2 = contextvars.copy_context()
    t1 = threading.Thread(target=ctx1.run, args=(session_run, "s1", "M1"))
    t2 = threading.Thread(target=ctx2.run, args=(session_run, "s2", "M2"))
    t1.start(); t2.start()
    t1.join(); t2.join()

    # each session must see its OWN manager every time, no cross-contamination
    assert all(name == "M1" for name in sessions["s1"])
    assert all(name == "M2" for name in sessions["s2"])
