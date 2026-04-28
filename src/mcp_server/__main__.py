"""Entry point: ``python -m src.mcp_server --workspace <name>``.

Parses CLI arguments, configures stderr logging (stdio is reserved for
the MCP transport — never log to stdout), and dispatches to
:func:`src.mcp_server.server.run_server`.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys


def main() -> int:
    """CLI entry point for the Sentinel MCP server.

    Returns:
        Process exit code: 0 on graceful shutdown.
    """
    parser = argparse.ArgumentParser(
        prog="sentinel-mcp",
        description=(
            "Sentinel MCP server — exposes Sentinel tools "
            "(get_dataset_schema, query_data, propose_state_change, "
            "query_past_incidents, get_pending_changes, "
            "commit_pending_changes) to external MCP clients."
        ),
    )
    parser.add_argument(
        "--workspace",
        required=True,
        help=(
            "Workspace name under data/workspaces/. Must already exist "
            "(create one by uploading a CSV in the Streamlit UI first)."
        ),
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity (default: INFO).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        # IMPORTANT: stderr only — stdout is the MCP stdio transport.
        stream=sys.stderr,
    )

    # Defer the import so ``--help`` works even if MCP/Sentinel imports
    # would fail (helpful when diagnosing install issues).
    from src.mcp_server.server import run_server

    asyncio.run(run_server(args.workspace))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
