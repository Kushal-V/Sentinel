"""MCP server package — exposes Sentinel tools to external agents.

This package provides a Model Context Protocol (MCP) server that wraps
Sentinel's existing tool functions (defined in ``src.tools.tool_registry``)
and serves them over stdio so external agents — Claude Desktop, IDE
agents, custom orchestrators — can drive a Sentinel workspace remotely.

Public surface
--------------
* :func:`run_server` — async entry-point used by ``__main__``.
* :func:`_build_server` — constructs the configured ``mcp.server.Server``
  for a named workspace; exposed for unit tests.
"""

from src.mcp_server.server import _build_server, run_server

__all__ = ["run_server", "_build_server"]
