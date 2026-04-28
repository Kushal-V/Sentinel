"""
src/agents/groq_recovery.py
===========================
Recovery helpers for Groq API quirks.

Groq's hosted Llama models occasionally emit XML-shaped tool calls
(``<function=name>{json}</function>``) inside the body of a 400 error
instead of returning a structured ``tool_calls`` block on the assistant
message. The orchestrator and the inter-agent ``ask_other_agent`` tool
both need to recover from this so the user-visible run does not abort
on a vendor parsing bug.

This module centralises the regex extraction so it can be unit-tested
in isolation. Previously the same parser was inlined inside
``src/tools/tool_registry.py``; the call sites were brittle and the
function had no test coverage.

Public API
----------
``parse_groq_xml_tool_call(error_message)`` returns either:

* ``{"tool_name": str, "arguments": dict}`` on a successful match
* ``None`` when the input is empty, has no recognisable XML tool call,
  or cannot be parsed at all (graceful failure — no exceptions
  propagate to callers).

Callers should treat the dict as opaque and read ``tool_name`` /
``arguments`` keys explicitly.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Optional

_LOG = logging.getLogger(__name__)

# Match either ``<function=name attrs>{body}</function>`` or
# ``<function=name{body}</function>`` (Groq has emitted both shapes).
# The two alternatives are kept separate so we know which capture
# group holds the name vs. the body for each variant.
_XML_TOOL_CALL_RE = re.compile(
    r"<function=(\w+)[^>]*>(.*?)</function>"
    r"|<function=(\w+)(.*?)</function>",
    re.DOTALL,
)

# Heuristic kwarg pattern for ``query_data``-style errors that come
# back as ``search_term="value"`` rather than valid JSON.
_SEARCH_TERM_KWARG_RE = re.compile(
    r'^.*?search_term\s*=\s*["\'](.*)["\'].*$'
)


def parse_groq_xml_tool_call(error_message: str) -> Optional[dict[str, Any]]:
    """Parse a Groq XML-format tool call out of an API error message.

    Args:
        error_message: The string body of the Groq exception (typically
            ``str(exc)`` from a ``BadRequestError`` raised by ChatGroq).

    Returns:
        On success, a dict with keys:

        * ``tool_name`` (``str``): the LangChain tool name Groq tried
          to invoke (e.g. ``"query_data"``, ``"propose_state_change"``).
        * ``arguments`` (``dict``): the parsed kwargs dict to pass into
          ``tool.invoke(input=...)``. Always a dict — never a string,
          tuple, or ``None``.

        ``None`` if:

        * ``error_message`` is empty / ``None``
        * No ``<function=...>`` block can be located
        * The body of the function call cannot be coerced into a dict
          and no fallback heuristic matched

    The function never raises — malformed XML, exotic JSON values, and
    unexpected types all funnel into a ``None`` return so callers can
    do ``if (parsed := parse_groq_xml_tool_call(...)) is not None``.
    """
    if not error_message:
        return None

    try:
        match = _XML_TOOL_CALL_RE.search(error_message)
    except Exception:  # noqa: BLE001 — defensive; re.search shouldn't fail
        _LOG.debug("Regex search failed unexpectedly", exc_info=True)
        return None

    if not match:
        return None

    tool_name = match.group(1) or match.group(3)
    args_str = (match.group(2) or match.group(4) or "").strip()

    if not tool_name:
        return None

    arguments: Optional[dict[str, Any]] = None

    # Path 1 — body is valid JSON object (the common case).
    try:
        decoded = json.loads(args_str) if args_str else {}
        if isinstance(decoded, dict):
            arguments = decoded
    except (json.JSONDecodeError, ValueError):
        arguments = None

    # Path 2 — kwarg-style body, e.g. ``search_term="SFT-HD"``.
    if arguments is None:
        kwarg_match = _SEARCH_TERM_KWARG_RE.match(args_str)
        if kwarg_match:
            arguments = {"search_term": kwarg_match.group(1).strip()}
        elif args_str:
            # Path 3 — bare string fallback (assume it was a search_term).
            arguments = {
                "search_term": args_str.replace('"', "").replace("'", "").strip()
            }
        else:
            arguments = {}

    # propose_state_change without a parsable dict body is unrecoverable,
    # but callers expect a dict back — return empty dict so the tool
    # can produce its own helpful error message.
    if tool_name == "propose_state_change" and not isinstance(arguments, dict):
        arguments = {}

    if not isinstance(arguments, dict):
        # Final guard: never return non-dict ``arguments``.
        return None

    return {"tool_name": tool_name, "arguments": arguments}
