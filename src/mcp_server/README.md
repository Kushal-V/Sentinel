# Sentinel MCP Server

Expose Sentinel's data tools over the **Model Context Protocol (MCP)** so
external agents — Claude Desktop, IDE agents, custom orchestrators — can
drive a Sentinel workspace remotely without going through the Streamlit UI.

## What this is

A thin wrapper around the existing tools in `src.tools.tool_registry`. It
does not modify any tool, agent, sandbox, or state-manager code. The
server boots a per-process `FactoryDataManager` rooted at the chosen
workspace and serves six tools over stdio.

## Tools exposed

| Tool | Description |
|------|-------------|
| `get_dataset_schema` | Inferred schema profile (primary key, constraint pairs, dtypes). |
| `query_data` | Case-insensitive substring search over inventory rows. |
| `propose_state_change` | Stage a sandbox-validated change for HITL approval. |
| `query_past_incidents` | TF-IDF semantic search over historical incidents. |
| `get_pending_changes` | List currently-staged changes. |
| `commit_pending_changes` | Commit ALL staged changes (re-validates each). |

## Install

```bash
uv sync
```

This pulls in `mcp>=1.0.0` (currently resolves to `mcp==1.27.x`) along
with the rest of Sentinel's dependencies.

## Run

```bash
# From the project root, with the venv active:
python -m src.mcp_server --workspace default

# Or via the console script:
sentinel-mcp --workspace default
```

The workspace must already exist under `data/workspaces/`. Create one by
uploading a CSV in the Streamlit UI first, or use the bundled `default`
workspace.

Logs go to **stderr** (stdout is reserved for the MCP stdio transport).
Add `--log-level DEBUG` for verbose tracing.

## Register with Claude Desktop

Add the following to your `claude_desktop_config.json` (location varies
by OS — see the Anthropic docs):

```json
{
  "mcpServers": {
    "sentinel": {
      "command": "uv",
      "args": [
        "--directory",
        "/absolute/path/to/Sentinel",
        "run",
        "python",
        "-m",
        "src.mcp_server",
        "--workspace",
        "default"
      ]
    }
  }
}
```

Restart Claude Desktop. The six Sentinel tools will appear in the MCP
tool tray.

## One server per workspace

Each MCP server process serves exactly one workspace — workspace name is
required and has no default that could leak data between tenants. Run
multiple servers for multiple workspaces, each in its own process.

## Limitations

- **Per-process workspace** — switch workspaces by restarting the server.
- **In-process sandbox** — `ShadowSandbox` validation runs in the same
  process as the server. There is no remote sandbox or distributed lock.
- **Two-phase commit is process-local** — staged changes live in the
  server's `FactoryDataManager`. A separate Streamlit instance pointing
  at the same workspace will see the on-disk inventory but not the
  in-memory pending queue until commit.
- **No secrets exposed** — environment variables (e.g., `GROQ_API_KEY`)
  are never forwarded through MCP responses; only tool outputs are.
- **stdio transport only** — HTTP/SSE transports are not configured here.
  Add them if you need remote (non-local) MCP clients.

## Verify

```bash
# Imports resolve:
python -c "from src.mcp_server.server import _build_server; print('ok')"

# Tests pass:
pytest tests/test_mcp_server.py -v
```
