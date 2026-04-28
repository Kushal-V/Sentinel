"""
tests/eval/fixtures/mock_responses.py
=====================================

Deterministic stand-ins for the LLM responses produced by the Sentinel
pipeline.  These fixtures let the eval harness run end-to-end logic
(scenario load -> route -> tool calls -> compare to expectations) without
touching the network.

Two layers are provided:

* :func:`mock_dispatch_route` — simulates the Dispatcher's
  ``with_structured_output(DispatchRoute)`` call.  Returns a plain ``dict``
  shaped like ``DispatchRoute`` (so callers can wrap it in the real Pydantic
  model if desired, but tests are not required to).
* :func:`mock_specialist_steps` — simulates the manual ReAct loop in
  ``AgentOrchestrator.run_specialist`` by returning a canned three-step tool
  call sequence: ``get_dataset_schema`` -> ``query_data`` -> ``propose_state_change``.

Both helpers are agent-roster agnostic: they key off the crisis ``type`` and
the agent ID string only, so swapping in a new tenant's agent identifiers
does not require code changes here.
"""

from __future__ import annotations

from typing import Any


# ---------------------------------------------------------------------------
# Crisis type -> default agent routing
# ---------------------------------------------------------------------------
#
# Kept intentionally simple and overridable.  Scenarios that need to test
# routing edge cases can supply ``trust_overrides`` in YAML which the runner
# applies *after* this default routing, exercising the real fallback logic.

_CRISIS_DEFAULT_AGENT: dict[str, str] = {
    "STOCK_LOW": "maker",
    "STOCK_OUT": "maker",
    "PRODUCTION_HALT": "maker",
    "SHIPMENT_DELAY": "mover",
    "ROUTE_DISRUPTION": "mover",
    "QUALITY_ISSUE": "keeper",
    "AUDIT_FLAG": "keeper",
}

_CRISIS_DEFAULT_URGENCY: dict[str, str] = {
    "STOCK_LOW": "MEDIUM",
    "STOCK_OUT": "CRITICAL",
    "PRODUCTION_HALT": "CRITICAL",
    "SHIPMENT_DELAY": "HIGH",
    "ROUTE_DISRUPTION": "HIGH",
    "QUALITY_ISSUE": "MEDIUM",
    "AUDIT_FLAG": "LOW",
}


def mock_dispatch_route(crisis: dict[str, Any]) -> dict[str, Any]:
    """Return a ``DispatchRoute``-shaped dict for the given crisis.

    Parameters
    ----------
    crisis:
        Raw crisis dict (as parsed from YAML).  Must contain a ``type`` key;
        all other keys are optional.

    Returns
    -------
    dict
        Keys: ``selected_agent``, ``delegation_justification``,
        ``urgency_tier``, ``required_lookups``, ``trust_override_applied``,
        ``original_llm_choice``.
    """
    crisis_type = (crisis or {}).get("type", "UNKNOWN")
    sku = (crisis or {}).get("sku")
    agent = _CRISIS_DEFAULT_AGENT.get(crisis_type, "maker")
    urgency = _CRISIS_DEFAULT_URGENCY.get(crisis_type, "MEDIUM")

    return {
        "selected_agent": agent,
        "delegation_justification": f"Mock route: {crisis_type} -> {agent}",
        "urgency_tier": urgency,
        "required_lookups": [sku] if sku else [],
        "trust_override_applied": False,
        "original_llm_choice": None,
    }


def mock_specialist_steps(agent_id: str, crisis: dict[str, Any]) -> list[dict[str, Any]]:
    """Return the canonical three-step tool call sequence for any specialist.

    The ordering mirrors the contract enforced in ``src/agents/prompts.py``:
    ``get_dataset_schema`` -> ``query_data`` -> ``propose_state_change``.

    Parameters
    ----------
    agent_id:
        Canonical agent ID (e.g., ``"maker"``).  Echoed back inside the
        justification so tests can assert which agent ran.
    crisis:
        Raw crisis dict.  Used to pick a sensible row key + delta.

    Returns
    -------
    list[dict]
        Each element has shape::

            {
                "tool": <tool_name>,
                "args": {...},
                "result": {"status": "ok"|"SAFE", ...},
            }
    """
    crisis = crisis or {}
    sku = crisis.get("sku") or "UNKNOWN_SKU"
    crisis_type = crisis.get("type", "UNKNOWN")

    # Delta direction depends on whether the crisis is a shortage or surplus.
    # This is illustrative only — the sandbox is what really validates bounds.
    delta = 100 if crisis_type in {"STOCK_LOW", "STOCK_OUT"} else -10

    return [
        {
            "tool": "get_dataset_schema",
            "args": {},
            "result": {
                "status": "ok",
                "primary_key": "sku",
                "constraints": [("current_stock", "max_capacity")],
            },
        },
        {
            "tool": "query_data",
            "args": {"search_term": sku},
            "result": {"status": "ok", "rows_found": 1, "sku": sku},
        },
        {
            "tool": "propose_state_change",
            "args": {
                "row_key": sku,
                "target_column": "current_stock",
                "delta": delta,
                "justification": (
                    f"[{agent_id}] Mock proposal for {crisis_type} on {sku}"
                ),
            },
            "result": {"status": "SAFE", "sandbox_approved": True},
        },
    ]
