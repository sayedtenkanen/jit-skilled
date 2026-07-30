"""Enables `python -m jitskilled <run|optimize> ...` in addition to the
submodule form `python -m jitskilled.run_pipeline ...` /
`python -m jitskilled.optimize ...` (both keep working).
"""
from __future__ import annotations

import sys

from . import optimize, run_pipeline


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] not in ("run", "optimize"):
        print("usage: python -m jitskilled <run|optimize> [args...]", file=sys.stderr)
        raise SystemExit(2)
    command, rest = sys.argv[1], sys.argv[2:]
    sys.argv = [f"jitskilled {command}", *rest]
    if command == "run":
        run_pipeline.main()
    else:
        optimize.main()


if __name__ == "__main__":
    main()
