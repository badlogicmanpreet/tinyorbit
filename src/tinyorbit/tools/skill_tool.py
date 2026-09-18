"""The Skill tool — pull a named instruction bundle into the conversation (Ch 8).

Invoking a skill is not an action on the world; it returns the skill's
instructions as the tool result, so they land in the model's context for the
next turn. That makes it read-only and concurrency-safe. The available skills
are baked into the schema (a stable enum, cache-friendly) and their bodies are
held on the tool instance, so a call is a pure lookup with no I/O.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from tinyorbit.tools.base import Tool, ToolResult, ToolUseContext

if TYPE_CHECKING:
    from tinyorbit.skills import Skill


class SkillTool(Tool):
    name = "Skill"

    def __init__(self, skills: list["Skill"] | None = None) -> None:
        self._skills = {s.name: s for s in (skills or [])}
        catalogue = "\n".join(f"- {s.name}: {s.description}" for s in self._skills.values())
        self.description = (
            "Load a named skill's instructions into the conversation. A skill is a "
            "packaged set of instructions for a specific task; invoke it when the task "
            "matches one below, then follow the instructions it returns.\n\n"
            "Available skills:\n" + (catalogue or "- (none)")
        )
        schema_type: dict = {"type": "string", "description": "Which skill to load."}
        if self._skills:
            schema_type["enum"] = list(self._skills)
        self.input_schema = {
            "type": "object",
            "properties": {"name": schema_type},
            "required": ["name"],
        }

    def is_read_only(self, inp: dict) -> bool:
        return True

    def is_concurrency_safe(self, inp: dict) -> bool:
        return True

    def describe_call(self, inp: dict) -> str:
        return f"Skill({inp.get('name', '?')})"

    async def call(self, inp: dict, ctx: ToolUseContext) -> ToolResult:
        skill = self._skills.get(inp["name"])
        if skill is None:
            available = ", ".join(self._skills) or "(none)"
            return ToolResult(
                f"unknown skill '{inp['name']}'. Available: {available}", is_error=True
            )
        header = f"# Skill: {skill.name}"
        if skill.description:
            header += f"\n{skill.description}"
        return ToolResult(f"{header}\n\n{skill.instructions}")
