"""
tests/eval/runner.py
====================

Loader + executor for declarative Sentinel evaluation scenarios.

Two public functions:

* :func:`load_scenarios` — read a YAML file of scenario dicts; tolerant of
  missing files (returns ``[]``) and empty files (returns ``[]``).
* :func:`run_scenario` — execute a single scenario against an orchestrator
  (mocked by default), collect the actual routing + tool-call trace, compare
  to the scenario's ``expectations`` block, and return a
  :class:`ScenarioResult`.

Mock mode is the default and the only fully implemented path in this phase
(skeleton).  Live mode is intentionally a stub — wiring up a real
``AgentOrchestrator`` with a real Groq key is deferred to the follow-up
DeepEval/Ragas integration so that this skeleton stays hermetic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

try:
    # Used only to read AGENT_FALLBACKS for trust override resolution.
    # Imported lazily so the harness still loads if src.* is unavailable
    # (e.g., during isolated unit-test collection).
    from src.core import config as _sentinel_config  # type: ignore
    _AGENT_FALLBACKS: dict[str, str] = dict(_sentinel_config.AGENT_FALLBACKS)
    _ROUTING_THRESHOLD: float = float(_sentinel_config.ROUTING_THRESHOLD)
except Exception:  # pragma: no cover - defensive fallback for partial envs
    _AGENT_FALLBACKS = {"maker": "mover", "mover": "keeper", "keeper": "maker"}
    _ROUTING_THRESHOLD = 0.50


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class ScenarioResult:
    """Outcome of running a single scenario.

    Attributes
    ----------
    name:
        Scenario name (echoed from the YAML entry).
    passed:
        ``True`` iff every expectation matched the actual result.
    failures:
        Human-readable strings describing each mismatched expectation.
    metrics:
        Optional bag of metric values (filled in by callers using
        :mod:`tests.eval.metrics`).
    actual:
        The full actual-result dict — selected agent, tool calls, sandbox
        verdict — useful when a metric needs more than the boolean ``passed``.
    """

    name: str
    passed: bool
    failures: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    actual: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


def load_scenarios(path: Path) -> list[dict]:
    """Read scenarios from a YAML file.

    Returns ``[]`` if the file does not exist or is empty.  Always returns a
    list — never ``None`` — so callers can iterate unconditionally.
    """
    path = Path(path)
    if not path.exists():
        return []
    raw = yaml.safe_load(path.read_text()) or []
    if not isinstance(raw, list):
        # YAML containing a single mapping at top level — wrap it.
        return [raw]
    return raw


# ---------------------------------------------------------------------------
# Mock-mode execution
# ---------------------------------------------------------------------------


def _resolve_route_with_trust(
    base_route: dict[str, Any],
    trust_overrides: dict[str, float] | None,
) -> dict[str, Any]:
    """Apply Sentinel's trust override logic to a (mocked) base route.

    Mirrors the override rule in ``AgentOrchestrator``: if the selected
    agent's trust score is strictly below ``ROUTING_THRESHOLD``, the route
    is reassigned to the agent's configured fallback in
    ``config.AGENT_FALLBACKS`` and ``trust_override_applied`` is set.
    """
    route = dict(base_route)
    if not trust_overrides:
        return route

    chosen = route.get("selected_agent")
    score = trust_overrides.get(chosen) if chosen else None
    if score is not None and score < _ROUTING_THRESHOLD:
        fallback = _AGENT_FALLBACKS.get(chosen, chosen)
        route["original_llm_choice"] = chosen
        route["selected_agent"] = fallback
        route["trust_override_applied"] = True
    return route


def _run_scenario_mock(scenario: dict[str, Any]) -> dict[str, Any]:
    """Execute the scenario entirely in-process using fixture stubs."""
    # Imported lazily to avoid a hard package-level dependency cycle.
    from .fixtures.mock_responses import (
        mock_dispatch_route,
        mock_specialist_steps,
    )

    crisis = scenario.get("crisis", {}) or {}
    trust_overrides = scenario.get("trust_overrides")

    base_route = mock_dispatch_route(crisis)
    route = _resolve_route_with_trust(base_route, trust_overrides)

    selected_agent = route["selected_agent"]
    steps = mock_specialist_steps(selected_agent, crisis)

    # Sandbox verdict in mock mode = AND of every step's reported approval.
    sandbox_approved = all(
        s.get("result", {}).get("status") in {"SAFE", "ok"}
        and s.get("result", {}).get("sandbox_approved", True)
        for s in steps
    )

    return {
        "route": route,
        "selected_agent": selected_agent,
        "trust_override_applied": route.get("trust_override_applied", False),
        "tool_calls": steps,
        "sandbox_approved": sandbox_approved,
    }


# ---------------------------------------------------------------------------
# Public runner
# ---------------------------------------------------------------------------


def run_scenario(
    scenario: dict[str, Any],
    orchestrator: Any = None,
    mock_mode: bool = True,
) -> ScenarioResult:
    """Execute ``scenario`` and return a :class:`ScenarioResult`.

    Parameters
    ----------
    scenario:
        A dict shaped like the entries in ``scenarios.yaml``.
    orchestrator:
        Live ``AgentOrchestrator`` instance, used only when ``mock_mode`` is
        ``False``.  Ignored in mock mode.
    mock_mode:
        Default ``True``.  In mock mode, the runner uses the fixtures in
        :mod:`tests.eval.fixtures.mock_responses` and never calls Groq.
        Live mode is a stub in this skeleton phase and returns an
        always-failing result with an explanatory failure string.
    """
    name = scenario.get("name", "<unnamed>")
    expectations = scenario.get("expectations", {}) or {}

    if not mock_mode:
        return ScenarioResult(
            name=name,
            passed=False,
            failures=["live mode not implemented in this skeleton phase"],
            actual={},
        )

    actual = _run_scenario_mock(scenario)

    # ------------------------------------------------------------------ #
    # Compare actual vs expected.  Each expectation is optional; absent  #
    # keys in expectations are simply not checked.                        #
    # ------------------------------------------------------------------ #
    failures: list[str] = []

    if "selected_agent" in expectations:
        if actual["selected_agent"] != expectations["selected_agent"]:
            failures.append(
                f"selected_agent: expected {expectations['selected_agent']!r}, "
                f"got {actual['selected_agent']!r}"
            )

    if "sandbox_approved" in expectations:
        if bool(actual["sandbox_approved"]) != bool(expectations["sandbox_approved"]):
            failures.append(
                f"sandbox_approved: expected {expectations['sandbox_approved']!r}, "
                f"got {actual['sandbox_approved']!r}"
            )

    if "trust_override_applied" in expectations:
        if bool(actual["trust_override_applied"]) != bool(
            expectations["trust_override_applied"]
        ):
            failures.append(
                "trust_override_applied: expected "
                f"{expectations['trust_override_applied']!r}, "
                f"got {actual['trust_override_applied']!r}"
            )

    if "min_tool_calls" in expectations:
        n = len(actual.get("tool_calls", []))
        if n < int(expectations["min_tool_calls"]):
            failures.append(
                f"min_tool_calls: expected >= {expectations['min_tool_calls']}, got {n}"
            )

    return ScenarioResult(
        name=name,
        passed=not failures,
        failures=failures,
        actual=actual,
    )
