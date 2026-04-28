"""MCP server exposing Sentinel data tools.

Wraps the existing tool functions in :mod:`src.tools.tool_registry` and
serves them via the Model Context Protocol (MCP). External agents
(Claude Desktop, IDE agents, custom orchestrators) can drive a Sentinel
workspace remotely without going through the Streamlit UI.

Tools exposed
-------------
* ``get_dataset_schema()`` — return the inferred schema profile.
* ``query_data(search_term)`` — read inventory rows by key/term.
* ``propose_state_change(...)`` — stage a sandbox-validated change.
* ``query_past_incidents(crisis_summary, top_k)`` — semantic incident search.
* ``get_pending_changes()`` — list currently-staged changes.
* ``commit_pending_changes()`` — commit ALL staged changes (HITL-equivalent).

Process model
-------------
Each MCP server boots a per-process :class:`FactoryDataManager` rooted at
the chosen workspace. Multiple servers for different workspaces run as
separate processes. The MCP server runs in a separate process from the
Streamlit app — there is no shared singleton conflict because each
process owns its own ``_data_manager``.

Wrapper discipline
------------------
This module is a *thin wrapper*: it does NOT modify any tool, agent,
sandbox, schema, or state-manager code. All validation (sandbox checks,
schema inference, two-phase commit re-validation) flows through the
existing functions unchanged.
"""

from __future__ import annotations

import logging
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from src.core.config import WORKSPACES_DIR
from src.core.state_manager import FactoryDataManager
from src.tools.tool_registry import (
    commit_pending_changes as _commit_pending_changes,
)
from src.tools.tool_registry import (
    discard_pending_changes as _discard_pending_changes,
)
from src.tools.tool_registry import (
    get_dataset_schema,
    propose_state_change,
    query_data,
    query_past_incidents,
    set_data_manager,
)

_LOG = logging.getLogger("sentinel-mcp")


def _build_server(workspace_name: str) -> Server:
    """Construct and configure an MCP server for the given workspace.

    Boots a :class:`FactoryDataManager` rooted at ``WORKSPACES_DIR /
    workspace_name`` and registers it as the process-wide tool data
    manager via :func:`set_data_manager`. Then registers six tools on
    a fresh ``mcp.server.Server`` instance and returns it.

    The function is exposed (rather than inlined into :func:`run_server`)
    so unit tests can verify the server builds correctly without having
    to spin up a stdio loop.

    Args:
        workspace_name: Name of an existing workspace under
            ``data/workspaces/``. Must already exist on disk.

    Returns:
        A fully-configured :class:`mcp.server.Server` ready to be passed
        to ``server.run(read_stream, write_stream, init_options)``.

    Raises:
        FileNotFoundError: If the workspace directory does not exist.
        ValueError: If the workspace name contains path-traversal
            characters (raised by :class:`FactoryDataManager`).
    """
    workspace_dir = WORKSPACES_DIR / workspace_name
    if not workspace_dir.exists():
        raise FileNotFoundError(
            f"Workspace not found: {workspace_dir}. "
            f"Create it first via the Streamlit UI or by uploading a CSV."
        )

    # FactoryDataManager takes a workspace NAME (not a path) and
    # internally resolves it to WORKSPACES_DIR / name. The constructor
    # also performs path-traversal validation on the name.
    manager = FactoryDataManager(workspace_name)
    set_data_manager(manager)

    server: Server = Server("sentinel-mcp")

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        """Advertise the six Sentinel tools to MCP clients."""
        return [
            Tool(
                name="get_dataset_schema",
                description=(
                    "Return the inferred schema profile for the active "
                    "workspace: primary key, constraint pairs, and column "
                    "types. Call this FIRST so you learn the column names."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            ),
            Tool(
                name="query_data",
                description=(
                    "Search inventory rows for a case-insensitive substring "
                    "match across all text columns. Returns matched rows as "
                    "JSON. Call this BEFORE proposing a change."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "search_term": {
                            "type": "string",
                            "description": (
                                "Keyword or partial value to search for; "
                                "case-insensitive substring match."
                            ),
                        },
                    },
                    "required": ["search_term"],
                },
            ),
            Tool(
                name="propose_state_change",
                description=(
                    "Stage a sandbox-validated state change for HITL "
                    "approval. Does NOT commit — call commit_pending_changes "
                    "to apply staged proposals to live state."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "row_key": {
                            "type": "string",
                            "description": (
                                "Exact primary key value of the row to "
                                "modify (e.g., 'ITEM-PLASTIC-01')."
                            ),
                        },
                        "target_column": {
                            "type": "string",
                            "description": (
                                "Name of the numeric column to modify "
                                "(must match a column in the schema)."
                            ),
                        },
                        "delta": {
                            "type": "number",
                            "description": (
                                "Signed change. Positive increases the "
                                "value; negative decreases it."
                            ),
                        },
                        "justification": {
                            "type": "string",
                            "description": (
                                "Brief explanation logged to the audit "
                                "ledger for the Analyst."
                            ),
                        },
                    },
                    "required": [
                        "row_key",
                        "target_column",
                        "delta",
                        "justification",
                    ],
                },
            ),
            Tool(
                name="query_past_incidents",
                description=(
                    "Semantic TF-IDF search over historical incidents in "
                    "this workspace. Returns up to top_k similar past "
                    "events with similarity scores."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "crisis_summary": {
                            "type": "string",
                            "description": (
                                "Short description of the current crisis "
                                "to find similar historical incidents for."
                            ),
                        },
                        "top_k": {
                            "type": "integer",
                            "description": (
                                "Number of incidents to return "
                                "(1-10, default 5)."
                            ),
                            "default": 5,
                        },
                    },
                    "required": ["crisis_summary"],
                },
            ),
            Tool(
                name="get_pending_changes",
                description=(
                    "Return all currently-staged but not-yet-committed "
                    "changes for this workspace."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            ),
            Tool(
                name="commit_pending_changes",
                description=(
                    "Commit ALL currently-staged changes via ShadowSandbox "
                    "re-validation. Equivalent to clicking Approve in the "
                    "Streamlit HITL panel. Stale (now-invalid) changes are "
                    "rejected and re-staged."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            ),
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
        """Dispatch tool calls to the underlying Sentinel functions.

        Each branch invokes the existing LangChain ``@tool`` via
        ``.invoke({...})`` (the supported call style for tool objects)
        or the plain function for non-tool helpers. Errors are caught
        and returned as text so the MCP client receives a useful message
        instead of a transport-level failure.
        """
        try:
            if name == "get_dataset_schema":
                result = get_dataset_schema.invoke({})
            elif name == "query_data":
                result = query_data.invoke(
                    {"search_term": arguments["search_term"]}
                )
            elif name == "propose_state_change":
                # The underlying tool reads ``config`` via LangChain's
                # RunnableConfig injection, so we pass the configurable
                # agent_id explicitly to identify MCP-driven changes in
                # the audit log.
                result = propose_state_change.invoke(
                    {
                        "row_key": arguments["row_key"],
                        "target_column": arguments["target_column"],
                        "delta": arguments["delta"],
                        "justification": arguments["justification"],
                    },
                    config={"configurable": {"agent_id": "mcp_client"}},
                )
            elif name == "query_past_incidents":
                result = query_past_incidents.invoke(
                    {
                        "crisis_summary": arguments["crisis_summary"],
                        "top_k": arguments.get("top_k", 5),
                    }
                )
            elif name == "get_pending_changes":
                staged = manager.get_pending_changes()
                if not staged:
                    result = "(no pending changes)"
                else:
                    lines = [f"{len(staged)} pending change(s):"]
                    for i, change in enumerate(staged, 1):
                        lines.append(
                            f"{i}. row={change.get('row_key')!r} "
                            f"col={change.get('target_column')!r} "
                            f"delta={change.get('delta')} "
                            f"old={change.get('old_value')} "
                            f"-> new={change.get('new_value')} "
                            f"| justification: {change.get('justification', '')}"
                        )
                    result = "\n".join(lines)
            elif name == "commit_pending_changes":
                committed = _commit_pending_changes()
                if not committed:
                    result = "Committed: 0 changes (queue empty or all rejected)."
                else:
                    summaries = [
                        f"{c['row_key']}.{c['target_column']} "
                        f"({c['old_value']}->{c['new_value']})"
                        for c in committed
                    ]
                    result = (
                        f"Committed {len(committed)} change(s): "
                        + "; ".join(summaries)
                    )
            elif name == "discard_pending_changes":
                # Not advertised in list_tools but available defensively;
                # exposing rejection lets external clients drop a stale
                # queue without needing UI access.
                count = _discard_pending_changes()
                result = f"Discarded {count} pending change(s)."
            else:
                result = f"ERROR: Unknown tool '{name}'"
        except KeyError as exc:
            _LOG.exception("Tool '%s' missing required argument: %s", name, exc)
            result = f"ERROR: Tool '{name}' missing required argument {exc}"
        except Exception as exc:  # noqa: BLE001
            _LOG.exception("Tool '%s' failed", name)
            result = f"ERROR: {exc}"
        return [TextContent(type="text", text=str(result))]

    _LOG.info(
        "MCP server built for workspace '%s' (path=%s).",
        workspace_name,
        workspace_dir,
    )
    return server


async def run_server(workspace_name: str) -> None:
    """Run the MCP server over stdio for the named workspace.

    This is the long-running async entrypoint invoked by
    :mod:`src.mcp_server.__main__`. It:

    1. Builds the :class:`mcp.server.Server` via :func:`_build_server`.
    2. Opens a stdio transport (the MCP standard for local agents).
    3. Drives ``server.run`` until the client disconnects or the
       process is terminated.

    Args:
        workspace_name: Name of the workspace to expose. Must exist
            on disk under ``data/workspaces/``.
    """
    server = _build_server(workspace_name)
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )
