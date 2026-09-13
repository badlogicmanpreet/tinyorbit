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
    initial_prompt: str | None = None
    print_mode: bool = False
    trusted: bool = False
    permission_mode: PermissionMode = PermissionMode.DEFAULT
    allow_rules: list[str] = field(default_factory=list)
    effort: str | None = None
    fallbacks: bool = True
    max_turns: int | None = None
    data_dir: Path = Path.home() / ".tinyorbit"
    capabilities: dict = field(default_factory=dict)


_init_cache: Config | None = None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tinyorbit", add_help=False)
    parser.add_argument("prompt", nargs="?", default=None)
    parser.add_argument("--print", dest="print_mode", action="store_true")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--cwd", default=os.getcwd())
    parser.add_argument("--permission-mode", default=PermissionMode.DEFAULT.value,
                        choices=[m.value for m in PermissionMode])
    parser.add_argument("--allow", action="append", default=[], metavar="RULE",
                        help="permission rule, e.g. 'Bash(git status*)' or 'Edit'")
    parser.add_argument("--effort", default=None, choices=["low", "medium", "high", "xhigh", "max"])
    parser.add_argument("--no-fallbacks", dest="fallbacks", action="store_false")
    parser.add_argument("--max-turns", type=int, default=None)
    return parser


def init(argv: list[str]) -> Config:
    """Phase 2. Parse args, resolve config, establish the trust boundary. Idempotent."""
    global _init_cache
    if _init_cache is not None:
        return _init_cache

    args = build_parser().parse_args(argv)
    config = Config(
        cwd=Path(args.cwd).resolve(),
        model=args.model,
        initial_prompt=args.prompt,
        print_mode=args.print_mode,
        permission_mode=PermissionMode(args.permission_mode),
        allow_rules=list(args.allow),
        effort=args.effort,
        fallbacks=args.fallbacks,
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
    from tinyorbit.memory import load_memory
    from tinyorbit.prompt import build_system_prompt
    from tinyorbit.tools import assemble_tool_pool

    tools = assemble_tool_pool()                                        # ch 6
    memory = load_memory(config.cwd) if config.trusted else []          # ch 11
    config.capabilities["tools"] = tools
    config.capabilities["memory"] = memory
    config.capabilities["system_prompt"] = build_system_prompt(config.cwd, memory)   # ch 4
    config.capabilities["hooks"] = []                                   # ch 12
    config.capabilities["agents"] = []                                  # ch 8
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
