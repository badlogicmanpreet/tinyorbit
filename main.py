"""CLI entry point for tinyorbit.

Built in Chapter 2 (Bootstrap). Mirrors the 5-phase startup pipeline:

    Phase 0 — fast-path dispatch (this file)
    Phase 1 — module-level I/O (Python doesn't have JS's module-eval trick;
              we use lazy imports below to keep the fast path light)
    Phase 2 — init()  : parse args, resolve config, trust boundary
    Phase 3 — setup() : register tools / hooks / agents
    Phase 4 — launch(): dispatch to REPL or --print mode

All paths eventually call the query loop (built in Ch 5).
"""

from __future__ import annotations

import sys

USAGE = """\
tinyorbit — a small terminal coding agent (learning project)

usage:
  tinyorbit [PROMPT]              start an interactive session (optionally with a first prompt)
  tinyorbit --print "QUESTION"    one-shot, headless. Streams the answer to stdout.
  tinyorbit --version             print version and exit
  tinyorbit --help                print this help and exit

flags:
  --model MODEL                  model id (default: claude-opus-5)
  --provider NAME                model dialect: anthropic | openai | <registered>
                                 (default: inferred from model / $TINYORBIT_PROVIDER)
  --cwd PATH                     working directory (default: current dir)
  --permission-mode MODE         default | acceptEdits | bypassPermissions
  --allow RULE                   pre-approve a tool, e.g. 'Bash(git status*)' (repeatable)
  --effort LEVEL                 low | medium | high | xhigh | max
  --max-turns N                  stop after N tool-use turns
  --no-fallbacks                 disable server-side refusal fallback

credentials: ANTHROPIC_API_KEY, or a profile from `ant auth login`.
"""


def fast_path(argv: list[str]) -> int | None:
    """Phase 0. Handle invocations that don't need the full bootstrap.

    Returns an exit code if we handled it, else None to fall through.
    The point: --version should NOT load the rest of the system.
    """
    if len(argv) >= 2 and argv[1] in ("--version", "-V"):
        # Lazy import keeps this path cheap.
        from tinyorbit import __version__
        print(f"tinyorbit {__version__}")
        return 0

    if len(argv) >= 2 and argv[1] in ("--help", "-h"):
        print(USAGE)
        return 0

    return None


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv

    rc = fast_path(argv)
    if rc is not None:
        return rc

    # Heavy imports start here, AFTER the fast path. This is Python's
    # rough equivalent of the JS "module-level I/O" trick — we just
    # don't pay the import cost on `--version` or `--help`.
    from tinyorbit.bootstrap import init, launch, setup

    # Phase 2: parse args, resolve config, trust boundary.
    config = init(argv[1:])

    # Phase 3: register capabilities.
    setup(config)

    # Phase 4: pick a launch path and run.
    return launch(config)


if __name__ == "__main__":
    sys.exit(main())
