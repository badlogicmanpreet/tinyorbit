"""Where MCP servers come from — env-driven, opt-in, and free when unused.

Two sources, both off by default so a plain run (and every benchmark run) pays
nothing and behaves identically:

  * TINYORBIT_MCP_CONFIG=<path>   — a `.mcp.json` of plain HTTP servers.
  * TINYORBIT_MCP_SOURCES=a,b     — named sources, each lazily imported by
                                    convention as `tinyorbit.mcp.sources.<name>`.

The named-source hook mirrors `resolve_provider`: no generic file names a
specific gateway; the name arrives at runtime and the module self-registers on
import. That is how an internal, transport-specific server list (mTLS, private
auth) plugs in without any core file referencing it.
"""

from __future__ import annotations

import importlib
import json
import os
import re
from pathlib import Path
from typing import Callable

from tinyorbit.mcp.config import McpServerConfig

# ── named-source registry (self-registered on lazy import) ─────────────────────

McpSourceFactory = Callable[[], list[McpServerConfig]]
_SOURCES: dict[str, McpSourceFactory] = {}


def register_mcp_source(name: str, factory: McpSourceFactory) -> None:
    _SOURCES[name] = factory


def resolve_mcp_source(name: str) -> list[McpServerConfig]:
    if name not in _SOURCES:
        try:
            importlib.import_module(f"tinyorbit.mcp.sources.{name}")
        except ImportError:
            pass
    factory = _SOURCES.get(name)
    return list(factory()) if factory else []


# ── generic .mcp.json parsing ──────────────────────────────────────────────────

_ENV_REF = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def expand_env(value: str) -> str:
    """Expand `${VAR}` and `${VAR:-default}` against the environment."""
    def repl(match: re.Match) -> str:
        var, default = match.group(1), match.group(2)
        return os.environ.get(var) or (default or "")
    return _ENV_REF.sub(repl, value)


def parse_mcp_json(data: dict) -> list[McpServerConfig]:
    """Read a `.mcp.json` mapping into HTTP server configs (others skipped).

    Accepts the common wrappers (`mcpServers` / `servers`) or a bare mapping of
    name -> entry. Only `type: http` (streamable-HTTP) is understood here.
    """
    servers = data.get("mcpServers") or data.get("servers") or data
    configs: list[McpServerConfig] = []
    for name, entry in servers.items():
        if not isinstance(entry, dict) or entry.get("type", "http") != "http":
            continue
        url = expand_env(entry.get("url", ""))
        if not url:
            continue
        headers = {k: expand_env(str(v)) for k, v in (entry.get("headers") or {}).items()}
        configs.append(McpServerConfig(name=name, url=url, headers=headers))
    return configs


def load_mcp_json(path: Path) -> list[McpServerConfig]:
    return parse_mcp_json(json.loads(Path(path).read_text()))


# ── the entry point bootstrap.setup() calls ────────────────────────────────────

def discover_mcp_servers() -> list[McpServerConfig]:
    """Assemble the configured server list. Empty unless env opts in."""
    configs: list[McpServerConfig] = []
    if path := os.environ.get("TINYORBIT_MCP_CONFIG"):
        configs += load_mcp_json(Path(path))
    for name in (os.environ.get("TINYORBIT_MCP_SOURCES") or "").split(","):
        name = name.strip()
        if name:
            configs += resolve_mcp_source(name)
    return configs
