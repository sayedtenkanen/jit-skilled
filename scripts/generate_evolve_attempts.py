#!/usr/bin/env python3
"""Regenerate the recorded zero-shot attempt + pass/fail label on every
row of data/evolve.jsonl.

Real SkillTTA's retrieval pool carries full past agent trajectories --
what was actually tried, and whether it worked -- not just the gold
answer. This script gives the "lite" evolve pool the same shape: for each
evolve question, it runs a deterministic zero-shot MockClient solve
against the source document, grades it, and stores the resulting
`prior_attempt` / `prior_label` fields alongside the existing
`question` / `ground_truth`. `prompts.py` then surfaces this in the
retrieved-examples block sent to skill synthesis, so a synthesized skill
can reference what was tried and whether it worked -- not only what the
correct answer turned out to be.

MockClient is deterministic, so this is idempotent (rerun any time the
dataset changes) and produces a reproducible pool without needing an API
key. It intentionally uses the same zero-shot heuristic regardless of
which backend end users pick for the real pipeline -- the recorded
attempts describe "what a naive zero-shot pass looked like on this
dataset once", not a live trace from whichever backend is configured.

Usage:
  python scripts/generate_evolve_attempts.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    from jitskilled.grader import grade
    from jitskilled.llm import MockClient

    documents = {
        p.stem: p.read_text() for p in (ROOT / "data" / "documents").glob("*.txt")
    }
    evolve_path = ROOT / "data" / "evolve.jsonl"
    rows = [json.loads(line) for line in evolve_path.read_text().splitlines() if line.strip()]

    llm = MockClient()
    pass_count = 0
    for row in rows:
        doc_text = documents[row["source_doc"]]
        attempt = llm.solve(row["question"], doc_text)
        passed = grade(attempt, row["ground_truth"])
        row["prior_attempt"] = attempt
        row["prior_label"] = "pass" if passed else "fail"
        pass_count += passed
        print(f"  {row['task_id']}: {'pass' if passed else 'fail'} "
              f"(attempt={attempt!r} gt={row['ground_truth']!r})")

    evolve_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    print(f"[generate_evolve_attempts] {pass_count}/{len(rows)} zero-shot "
          f"attempts passed -> {evolve_path}")


if __name__ == "__main__":
    main()
