"""The Task tool — dispatch a subagent (Ch 8).

Running a subagent is not a new mechanism; it is the *same* agent loop invoked
with a fresh context. So this tool is thin: pick the named agent, build its
message history (one user turn: the task), build its tool pool (the parent's
tools minus Task — a subagent cannot spawn subagents, which is what stops
infinite recursion), and drive query() to completion. Only the final assistant
text returns to the parent; the child's tool calls stay in the child.

The tool needs a provider, a model and a cost ledger, none of which live in
ToolUseContext by default — a plain file tool never needs them. They arrive via
`ctx.subagent` (a SubagentContext), attached by the REPL where the provider is
built. When it is absent — under a unit test that never wired subagents, or
inside a subagent itself — the tool fails closed with a clear message rather
than pretending to work.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

from tinyorbit.tools.base import Tool, ToolResult, ToolUseContext

if TYPE_CHECKING:
    from tinyorbit.agents import AgentDefinition


class TaskTool(Tool):
    name = "Task"

    def __init__(self, agents: list["AgentDefinition"] | None = None) -> None:
        # The agent roster is baked into the schema at build time (a stable
        # enum keeps the tool description cache-friendly and steers the model
        # toward valid names). The runtime roster still comes from ctx, so a
        # mismatch is validated there too.
        names = [a.name for a in (agents or [])]
        descriptions = "\n".join(f"- {a.name}: {a.description}" for a in (agents or []))
        self.description = (
            "Dispatch a task to a specialized subagent that runs in its own context "
            "and returns only its final result. Use this to delegate self-contained "
            "work (research, a focused search, a scoped change) without filling the "
            "main conversation with intermediate steps. The subagent cannot dispatch "
            "further subagents.\n\nAvailable agents:\n" + (descriptions or "- (none)")
        )
        schema_type: dict = {"type": "string", "description": "Which subagent to run."}
        if names:
            schema_type["enum"] = names
        self.input_schema = {
            "type": "object",
            "properties": {
                "subagent_type": schema_type,
                "description": {
                    "type": "string",
                    "description": "A short (3-5 word) label for the task.",
                },
                "prompt": {
                    "type": "string",
                    "description": "The full task for the subagent. It sees only this, "
                                   "not the parent conversation, so include all needed context.",
                },
            },
            "required": ["subagent_type", "prompt"],
        }

    # A subagent may run writes/bash; treat dispatch as a non-read-only, serial
    # action. The child's own tool calls are still permission-checked one by one.
    def describe_call(self, inp: dict) -> str:
        label = inp.get("description") or inp.get("subagent_type", "?")
        return f"Task({inp.get('subagent_type', '?')}): {label}"

    def validate(self, inp: dict) -> str | None:
        if not str(inp.get("prompt", "")).strip():
            return "prompt must be non-empty"
        return None

    async def call(self, inp: dict, ctx: ToolUseContext) -> ToolResult:
        sc = ctx.subagent
        if sc is None:
            return ToolResult("subagents are not available in this context", is_error=True)

        agent = sc.find(inp["subagent_type"])
        if agent is None:
            available = ", ".join(a.name for a in sc.agents) or "(none)"
            return ToolResult(
                f"unknown subagent '{inp['subagent_type']}'. Available: {available}",
                is_error=True,
            )

        # Lazy import: query imports the tools package, so importing query at
        # module load would close an import cycle. Deferring it to call time
        # (subagents are opt-in and rare) keeps the base tool package clean.
        from tinyorbit.query import AssistantTurn, Done, QueryParams, query

        child_tools = self._child_tools(sc, agent)
        # Fresh context: no subagent handle (blocks recursion) and a clean
        # staleness ledger, but the same cwd, abort event and permission policy
        # — an interrupt still cancels the child, and its writes still prompt.
        child_ctx = replace(ctx, subagent=None, file_state={})

        params = QueryParams(
            messages=[{"role": "user", "content": inp["prompt"]}],
            system=[{"type": "text", "text": agent.prompt}],
            tools=child_tools,
            ctx=child_ctx,
            model=agent.model or sc.config.model,
            provider=sc.provider,
            cost=sc.cost,
            source=f"agent:{agent.name}",
            # Per-subagent budget wins over the parent loop's; None falls back to it.
            max_turns=agent.max_turns if agent.max_turns is not None else getattr(sc.config, "max_turns", None),
            effort=agent.effort or getattr(sc.config, "effort", None),
            fallbacks=getattr(sc.config, "fallbacks", True),
            thinking_display=getattr(sc.config, "thinking_display", "omitted"),
        )

        final_text = ""
        reason = "completed"
        async for event in query(params, sc.deps):
            if isinstance(event, AssistantTurn):
                final_text = "".join(
                    b.text for b in event.message.content
                    if getattr(b, "type", "") == "text"
                )
            elif isinstance(event, Done):
                reason = event.reason

        if reason == "completed":
            return ToolResult(final_text or "(subagent returned no text)")
        # Surface a non-clean exit so the parent model can react rather than
        # trusting a truncated or aborted answer.
        return ToolResult(
            f"subagent '{agent.name}' ended early ({reason}).\n{final_text}".rstrip(),
            is_error=reason not in ("max_turns",),
        )

    @staticmethod
    def _child_tools(sc, agent: "AgentDefinition") -> list[Tool]:
        pool = list(sc.config.capabilities.get("tools", []))
        # Never hand a subagent the Task tool: that is what bounds recursion.
        pool = [t for t in pool if t.name != "Task"]
        if agent.tools is None:
            return pool
        allow = set(agent.tools)
        return [t for t in pool if t.name in allow]
