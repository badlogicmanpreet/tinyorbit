"""Skill discovery and the Skill tool — network-free."""

from __future__ import annotations

import asyncio

from tinyorbit.permissions import PermissionMode, PermissionPolicy
from tinyorbit.skills import Skill, discover_skills, parse_skill_file
from tinyorbit.tools import ToolUseContext
from tinyorbit.tools.skill_tool import SkillTool


def ctx(tmp_path):
    return ToolUseContext(
        cwd=tmp_path, abort=asyncio.Event(),
        permissions=PermissionPolicy(mode=PermissionMode.BYPASS), data_dir=tmp_path / ".d",
    )


# ── discovery ─────────────────────────────────────────────────────────────────

def test_discover_empty_without_dir(tmp_path):
    assert discover_skills(tmp_path) == []


def test_discover_skill_md_layout(tmp_path):
    d = tmp_path / ".tinyorbit" / "skills" / "deploy"
    d.mkdir(parents=True)
    (d / "SKILL.md").write_text(
        "---\nname: deploy\ndescription: ship it safely\n---\n1. run tests\n2. tag\n"
    )
    skills = discover_skills(tmp_path)
    assert len(skills) == 1
    assert skills[0].name == "deploy"
    assert skills[0].description == "ship it safely"
    assert "run tests" in skills[0].instructions


def test_discover_flat_layout(tmp_path):
    d = tmp_path / ".tinyorbit" / "skills"
    d.mkdir(parents=True)
    (d / "review.md").write_text("Check for tests, then style.")
    skills = discover_skills(tmp_path)
    assert len(skills) == 1
    assert skills[0].name == "review"          # falls back to the file stem
    assert skills[0].instructions == "Check for tests, then style."


def test_directory_skill_not_shadowed_by_flat(tmp_path):
    root = tmp_path / ".tinyorbit" / "skills"
    (root / "dup").mkdir(parents=True)
    (root / "dup" / "SKILL.md").write_text("---\nname: dup\n---\nfrom directory\n")
    (root / "dup.md").write_text("---\nname: dup\n---\nfrom flat file\n")
    skills = discover_skills(tmp_path)
    assert len(skills) == 1
    assert skills[0].instructions == "from directory"


def test_empty_body_skipped(tmp_path):
    f = tmp_path / "empty.md"
    f.write_text("---\nname: x\n---\n")
    assert parse_skill_file(f, "empty") is None


# ── the tool ───────────────────────────────────────────────────────────────────

def test_schema_lists_skills():
    t = SkillTool([Skill("deploy", "ship it", "steps")])
    assert t.name == "Skill"
    assert t.input_schema["properties"]["name"]["enum"] == ["deploy"]
    assert "deploy: ship it" in t.description
    assert t.is_read_only({}) and t.is_concurrency_safe({})


async def test_call_returns_instructions(tmp_path):
    t = SkillTool([Skill("deploy", "ship it", "1. run tests\n2. tag")])
    res = await t.call({"name": "deploy"}, ctx(tmp_path))
    assert not res.is_error
    assert "# Skill: deploy" in res.data
    assert "ship it" in res.data and "run tests" in res.data


async def test_unknown_skill_is_error(tmp_path):
    t = SkillTool([Skill("deploy", "ship it", "steps")])
    res = await t.call({"name": "ghost"}, ctx(tmp_path))
    assert res.is_error and "unknown skill 'ghost'" in res.data


def test_no_skills_schema_has_no_enum():
    t = SkillTool([])
    assert "enum" not in t.input_schema["properties"]["name"]
    assert "- (none)" in t.description
