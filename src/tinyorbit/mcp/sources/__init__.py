"""Named MCP server sources, imported on demand by `resolve_mcp_source`.

Each module registers a factory via `register_mcp_source(name, ...)` at import
time. Nothing here is imported eagerly; the core reaches a source only when
`TINYORBIT_MCP_SOURCES` asks for it by name (mirrors provider discovery).
"""
