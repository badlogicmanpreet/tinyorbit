"""Subagent definitions and discovery (Ch 8).

A subagent is the same agent loop run in a *fresh* context: its own system
prompt, its own (usually narrower) tool set, and a clean message history. The
parent dispatches a task, the child runs to completion, and only its final
answer comes back — the child's intermediate tool churn never enters the
parent's history. That isolation is the whole point: it keeps a long research
or search detour from bloating the main context.

Two pieces live here, both dependency-free so the open-source core stays clean:

    AgentDefinition   what a subagent IS — name, description, system prompt,
                      allowed tools, optional model override.
    discover_agents   where they come from — Markdown files with YAML-ish
                      frontmatter under `.tinyorbit/agents/`, parsed by hand
                      (no yaml dependency, same spirit as the MCP env parser).

The runtime handle a dispatching tool needs (provider, model, cost) is
`SubagentContext`; it is attached to `ToolUseContext.subagent` by whoever owns
the provider (the REPL), never by generic code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tinyorbit.frontmatter import parse_frontmatter


@dataclass(frozen=True)
class AgentDefinition:
    """A named subagent. `tools=None` means 'inherit every tool except Task'.

    `effort` and `max_turns` are per-subagent budget overrides: `None` means
    'inherit the parent loop's setting'. A bounded forensic subagent
    (`max_turns=40`, `effort=medium`) does a different amount of work than one
    that inherits a deep main-loop budget, which is why the override matters.
    """

    name: str
    description: str
    prompt: str
    tools: list[str] | None = None
    model: str | None = None
    effort: str | None = None
    max_turns: int | None = None


@dataclass
class SubagentContext:
    """The runtime a dispatching tool needs to spin up a nested query().

    Built by the REPL (where the provider and cost ledger live), attached to
    ToolUseContext.subagent, and read by TaskTool at call time. `deps` is the
    same QueryDeps injection seam the top-level loop uses, so tests can drive a
    subagent with a scripted model and no network.
    """

    config: Any                                  # Config: model, capabilities, effort, ...
    provider: Any                                # a Provider instance
    cost: Any                                    # CostTracker, shared so sub-costs roll up
    agents: list[AgentDefinition] = field(default_factory=list)
    deps: Any = None                             # QueryDeps | None

    def find(self, name: str) -> AgentDefinition | None:
        return next((a for a in self.agents if a.name == name), None)


# ── discovery ────────────────────────────────────────────────────────────────

def parse_agent_file(path: Path) -> AgentDefinition | None:
    """One `.md` file -> one AgentDefinition. None if it has no usable prompt."""
    meta, body = parse_frontmatter(path.read_text())
    prompt = body.strip()
    if not prompt:
        return None
    tools_raw = meta.get("tools", "").strip()
    tools = [t.strip() for t in tools_raw.split(",") if t.strip()] if tools_raw else None
    max_turns_raw = meta.get("max_turns", "").strip()
    max_turns = int(max_turns_raw) if max_turns_raw.isdigit() else None
    return AgentDefinition(
        name=meta.get("name") or path.stem,
        description=meta.get("description", ""),
        prompt=prompt,
        tools=tools,
        model=meta.get("model") or None,
        effort=meta.get("effort") or None,
        max_turns=max_turns,
    )


def discover_agents(cwd: Path) -> list[AgentDefinition]:
    """Load subagents from `<cwd>/.tinyorbit/agents/*.md`.

    Empty (the common, zero-cost case) unless the directory exists — so a repo
    with no subagents configured pays nothing and the Task tool never appears.
    """
    root = Path(cwd) / ".tinyorbit" / "agents"
    if not root.is_dir():
        return []
    out: list[AgentDefinition] = []
    for path in sorted(root.glob("*.md")):
        try:
            agent = parse_agent_file(path)
        except OSError:
            continue
        if agent is not None:
            out.append(agent)
    return out
