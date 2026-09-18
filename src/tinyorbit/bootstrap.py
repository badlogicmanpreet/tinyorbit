"""Bootstrap — the startup pipeline (Ch 2).

    init(argv)   — parse args, resolve config, hit the trust boundary.
                   Memoized: safe to call from multiple entry points.
    setup(cfg)   — register capabilities (tools, memory, system prompt).
    launch(cfg)  — dispatch to a mode. Both modes call the query loop.

Each function narrows the system's scope:
    init()   says "I know my configuration."
    setup()  says "I have all my capabilities."
    launch() says "I know which mode I'm running."
By the time the query loop runs, every decision has been made.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field
from pathlib import Path

from tinyorbit.permissions import PermissionMode

DEFAULT_MODEL = "claude-opus-5"


@dataclass
class Config:
    """The bootstrap output. Everything downstream reads this and never mutates it."""

    cwd: Path
    model: str
    provider: str | None = None     # None → resolve from $TINYORBIT_PROVIDER / model id
    initial_prompt: str | None = None
    print_mode: bool = False
    trusted: bool = False
    permission_mode: PermissionMode = PermissionMode.DEFAULT
    allow_rules: list[str] = field(default_factory=list)
    # Which tools are advertised to the model at all (fnmatch globs over tool
    # names; deny wins). Empty allow = every tool. Distinct from allow_rules,
    # which govern permission prompts for tools that ARE present.
    allowed_tools: list[str] = field(default_factory=list)
    disallowed_tools: list[str] = field(default_factory=list)
    # Session continuity: resume reopens a saved id and appends to it; fork loads
    # that id's history but writes under a fresh id, leaving the original intact.
    resume_session: str | None = None
    fork_session: bool = False
    effort: str | None = None
    fallbacks: bool = True
    thinking_display: str = "omitted"
    max_turns: int | None = None
    data_dir: Path = Path.home() / ".tinyorbit"
    capabilities: dict = field(default_factory=dict)


_init_cache: Config | None = None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tinyorbit", add_help=False)
    parser.add_argument("prompt", nargs="?", default=None)
    parser.add_argument("--print", dest="print_mode", action="store_true")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--provider", default=None,
                        help="model dialect: anthropic | openai | <registered>; default inferred from model")
    parser.add_argument("--cwd", default=os.getcwd())
    parser.add_argument("--permission-mode", default=PermissionMode.DEFAULT.value,
                        choices=[m.value for m in PermissionMode])
    parser.add_argument("--allow", action="append", default=[], metavar="RULE",
                        help="permission rule, e.g. 'Bash(git status*)' or 'Edit'")
    parser.add_argument("--allowed-tools", action="append", default=[], metavar="GLOB",
                        help="only advertise tools whose name matches (repeatable, comma-ok), "
                             "e.g. 'Read,Grep' or 'mcp__gw__*'")
    parser.add_argument("--disallowed-tools", action="append", default=[], metavar="GLOB",
                        help="hide tools whose name matches (repeatable, comma-ok); wins over allow")
    parser.add_argument("--resume", dest="resume_session", default=None, metavar="ID",
                        help="reopen a saved session by id and keep appending to it")
    parser.add_argument("--fork", dest="fork_session", action="store_true",
                        help="with --resume, branch that session into a new id (original untouched)")
    parser.add_argument("--effort", default=None, choices=["low", "medium", "high", "xhigh", "max"])
    parser.add_argument("--no-fallbacks", dest="fallbacks", action="store_false")
    parser.add_argument("--thinking-display", default="omitted", choices=["omitted", "summarized"],
                        help="'summarized' streams the model's reasoning summary; default omits it")
    parser.add_argument("--max-turns", type=int, default=None)
    return parser


def _tool_globs(cli_values: list[str], env_var: str) -> list[str]:
    """Flatten repeatable, comma-joinable tool globs; CLI wins, else env."""
    raw = cli_values if cli_values else [os.environ.get(env_var, "")]
    return [g.strip() for entry in raw for g in entry.split(",") if g.strip()]


def init(argv: list[str]) -> Config:
    """Phase 2. Parse args, resolve config, establish the trust boundary. Idempotent."""
    global _init_cache
    if _init_cache is not None:
        return _init_cache

    args = build_parser().parse_args(argv)
    config = Config(
        cwd=Path(args.cwd).resolve(),
        model=args.model,
        provider=args.provider,
        initial_prompt=args.prompt,
        print_mode=args.print_mode,
        permission_mode=PermissionMode(args.permission_mode),
        allow_rules=list(args.allow),
        # CLI flags take precedence; fall back to env so a benchmark/embedder can
        # constrain the pool without rewriting argv. Each entry may be comma-joined.
        allowed_tools=_tool_globs(args.allowed_tools, "TINYORBIT_ALLOWED_TOOLS"),
        disallowed_tools=_tool_globs(args.disallowed_tools, "TINYORBIT_DISALLOWED_TOOLS"),
        resume_session=args.resume_session,
        fork_session=args.fork_session,
        effort=args.effort,
        fallbacks=args.fallbacks,
        thinking_display=args.thinking_display,
        max_turns=args.max_turns,
    )

    # ── TRUST BOUNDARY ──
    # Above this line we read nothing but argv. Below it, with trust granted,
    # we read files from the working directory (AGENTS.md) and run commands.
    config.trusted = check_trust(config.cwd)
    if not config.trusted:
        # Untrusted directories get no memory and no auto-allowed writes.
        config.permission_mode = PermissionMode.DEFAULT

    _init_cache = config
    return config


def check_trust(cwd: Path) -> bool:
    """A production agent persists the answer to a trust file and prompts when unknown.

    We auto-trust; the boundary is what matters, and it is a real gate that
    decides whether setup() may read project files.
    """
    return True


def setup(config: Config) -> None:
    """Phase 3. Register everything the runtime needs. Each slot maps to a chapter."""
    from tinyorbit.agents import discover_agents
    from tinyorbit.hooks import discover_hooks
    from tinyorbit.mcp import discover_mcp_servers
    from tinyorbit.memory import load_memory
    from tinyorbit.prompt import build_system_prompt
    from tinyorbit.skills import discover_skills
    from tinyorbit.tools import assemble_tool_pool
    from tinyorbit.tools.skill_tool import SkillTool
    from tinyorbit.tools.task_tool import TaskTool

    tools = assemble_tool_pool()                                        # ch 6
    memory = load_memory(config.cwd) if config.trusted else []          # ch 11
    # Subagents and skills are opt-in: discovered from `.tinyorbit/`, empty in a
    # plain repo. Their tools (Task, Skill) only join the pool when something is
    # defined, so the default tool count and prompt cache are untouched.
    agents = discover_agents(config.cwd) if config.trusted else []      # ch 8
    skills = discover_skills(config.cwd) if config.trusted else []      # ch 8
    if agents:
        tools = tools + [TaskTool(agents)]
    if skills:
        tools = tools + [SkillTool(skills)]
    config.capabilities["tools"] = tools
    config.capabilities["memory"] = memory
    config.capabilities["skills"] = skills                              # ch 8
    config.capabilities["system_prompt"] = build_system_prompt(config.cwd, memory)   # ch 4
    # PreToolUse guard hooks: empty unless TINYORBIT_HOOKS opts in (an embedder
    # driving tinyorbit as a library sets this slot directly instead). ch 12
    config.capabilities["hooks"] = discover_hooks()                     # ch 12
    config.capabilities["agents"] = agents                              # ch 8
    # MCP tools are resolved here but connected in the REPL's async startup
    # (discovery is async; this is sync). Empty unless env opts in — a plain
    # run pays nothing.
    config.capabilities["mcp_servers"] = discover_mcp_servers()         # ch 15
    config.data_dir.mkdir(parents=True, exist_ok=True)


def launch(config: Config) -> int:
    """Phase 4. Pick a mode, run. Both modes converge on query()."""
    from tinyorbit.repl import run_interactive, run_print_mode

    if config.print_mode:
        return run_print_mode(config)
    return run_interactive(config)


def _reset_for_tests() -> None:
    global _init_cache
    _init_cache = None
