"""Offline slot-library optimization: compare two runs, bucket by
transition, run a critic per sampled case (shared across candidates), then
run --num_candidates independent editor proposals ("beam search" over
possible slot-library edits) and write each as its own candidate file so
you can pick the best one manually.

Usage:
  python -m jitskilled.optimize \
      --current_run runs/v1 --previous_run runs/zero_shot \
      --slot_library configs/slots_v1.yaml --output_prefix configs/slots_v2 \
      --num_candidates 3

Writes configs/slots_v2_candidate1.yaml, _candidate2.yaml, _candidate3.yaml,
plus configs/slots_v2_summary.json with the shared critic results and every
candidate's patch for comparison.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from .llm import get_client
from .slots import apply_patch, load_slots, render_slots, save_slots

ROOT = Path(__file__).resolve().parent.parent.parent


def _load_run(run_dir: Path) -> dict[str, dict]:
    traj_path = run_dir / "trajectories.jsonl"
    rows = {}
    with open(traj_path) as f:
        for line in f:
            row = json.loads(line)
            rows[row["task_id"]] = row
    return rows


def _bucket(current: dict[str, dict], previous: dict[str, dict]) -> dict[str, str]:
    buckets = {}
    for task_id, cur_row in current.items():
        prev_row = previous.get(task_id)
        cur_pass = cur_row["label"] == "pass"
        prev_pass = prev_row["label"] == "pass" if prev_row else None
        if prev_pass is None:
            continue
        if prev_pass and not cur_pass:
            buckets[task_id] = "regression"
        elif not prev_pass and cur_pass:
            buckets[task_id] = "improvement"
        elif prev_pass and cur_pass:
            buckets[task_id] = "persistent_success"
        else:
            buckets[task_id] = "persistent_failure"
    return buckets


_DEFAULT_WEIGHTS = {
    "regression": 3, "persistent_failure": 2,
    "improvement": 2, "persistent_success": 1,
}


def _sample_cases(buckets: dict[str, str], max_cases: int, seed: int) -> list[str]:
    if max_cases <= 0:
        return []
    if max_cases >= len(buckets):
        return list(buckets)
    rng = random.Random(seed)
    by_bucket: dict[str, list[str]] = {}
    for task_id, label in buckets.items():
        by_bucket.setdefault(label, []).append(task_id)
    total_weight = sum(_DEFAULT_WEIGHTS.get(b, 1) * len(ids) for b, ids in by_bucket.items())
    sampled: list[str] = []
    for label, ids in by_bucket.items():
        share = _DEFAULT_WEIGHTS.get(label, 1) * len(ids) / total_weight
        quota = max(1, round(share * max_cases))
        rng.shuffle(ids)
        sampled.extend(ids[:quota])
    return sampled[:max_cases]


def _build_case(task_id: str, transition: str, current: dict, previous: dict,
                 pool_by_id: dict[str, dict], current_slots: dict) -> dict:
    cur_row = current[task_id]
    prev_row = previous.get(task_id, {})
    retrieved = [
        {"task_id": rid, **{k: v for k, v in pool_by_id.get(rid, {}).items() if k != "task_id"}}
        for rid in cur_row.get("retrieved_task_ids", [])
    ]
    return {
        "task_id": task_id,
        "transition": transition,
        "outcome": "failed" if cur_row["label"] == "fail" else "passed",
        "task_context": cur_row["question"],
        "current_run": {
            "answer": cur_row["answer"],
            "ground_truth": cur_row["ground_truth"],
            "label": cur_row["label"],
        },
        "previous_run": {
            "answer": prev_row.get("answer"),
            "label": prev_row.get("label"),
        } if transition in ("regression", "improvement") else {},
        "skill_retrieval": {"retrieved_examples": retrieved},
        "current_slot_library": current_slots,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--current_run", required=True)
    parser.add_argument("--previous_run", required=True)
    parser.add_argument("--slot_library", required=True)
    parser.add_argument("--output_prefix", required=True,
                         help="Candidates are written to "
                              "<prefix>_candidate<N>.yaml and the shared "
                              "summary to <prefix>_summary.json.")
    parser.add_argument("--num_candidates", type=int, default=3,
                         help="Number of independent editor proposals to "
                              "generate from the same (shared) critic "
                              "results. Default: 3.")
    parser.add_argument("--max_cases", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--llm", choices=["anthropic", "ollama", "apple", "mock"], default=None,
        help="LLM backend. Default: auto-detect from ANTHROPIC_API_KEY / "
             "JITSKILLED_LLM_BACKEND env var, falling back to mock.",
    )
    args = parser.parse_args()

    llm = get_client(args.llm)
    print(f"[optimize] using {llm.__class__.__name__}")

    current = _load_run(Path(args.current_run))
    previous = _load_run(Path(args.previous_run))
    with open(ROOT / "data" / "evolve.jsonl") as f:
        pool_by_id = {row["task_id"]: row for row in map(json.loads, f)}
    current_slots = load_slots(args.slot_library)

    buckets = _bucket(current, previous)
    transition_counts = {}
    for label in buckets.values():
        transition_counts[label] = transition_counts.get(label, 0) + 1
    print(f"[optimize] transitions: {transition_counts}")
    if not buckets:
        print("[optimize] warning: no overlapping task_ids between "
              "--current_run and --previous_run; nothing to compare.")

    sampled_ids = _sample_cases(buckets, args.max_cases, args.seed)
    print(f"[optimize] sampled {len(sampled_ids)} cases for critique")

    critic_results = []
    for task_id in sampled_ids:
        case = _build_case(task_id, buckets[task_id], current, previous,
                            pool_by_id, current_slots)
        result = llm.critic(case)
        critic_results.append(result)
        print(f"  critic[{task_id}] ({buckets[task_id]}): "
              f"lesson={result.get('lesson')!r}")

    # Critic results are shared across every candidate (that's the expensive,
    # per-case reasoning step); each candidate gets its own independent
    # editor call over those same results, so N candidates cost N editor
    # calls but only one round of critique, not N.
    candidates = []
    for candidate_index in range(1, args.num_candidates + 1):
        editor_payload = {
            "slot_schema": {"categories": ["input", "output"]},
            "current_slot_library": current_slots,
            "current_slot_library_rendered": render_slots(current_slots),
            "run_summary": {
                "transition_counts": transition_counts,
                "current_run": args.current_run,
                "previous_run": args.previous_run,
            },
            "case_critic_results": critic_results,
            "candidate_index": candidate_index,
            "num_candidates": args.num_candidates,
            "instructions": (
                "Propose a DIFFERENT edit than you would for other candidate "
                "indices -- explore a distinct hypothesis about what would "
                "improve the slot library, not a minor rewording of the same "
                "idea." if args.num_candidates > 1 else None
            ),
        }
        patch = llm.editor(editor_payload)
        ops = patch.get("operations", [])
        print(f"[optimize] candidate {candidate_index}: {len(ops)} operation(s)")
        for op in ops:
            print(f"    {op['operation']} [{op['category']}] {op['slot_id']}: {op.get('reason')}")

        new_slots = apply_patch(current_slots, ops)
        output_path = Path(f"{args.output_prefix}_candidate{candidate_index}.yaml")
        save_slots(output_path, new_slots)
        candidates.append({
            "candidate_index": candidate_index,
            "output_path": str(output_path),
            "editor_patch": patch,
        })

    summary_path = Path(f"{args.output_prefix}_summary.json")
    with open(summary_path, "w") as f:
        json.dump({
            "transition_counts": transition_counts,
            "sampled_task_ids": sampled_ids,
            "critic_results": critic_results,
            "candidates": candidates,
        }, f, indent=2)

    candidate_paths = ", ".join(c["output_path"] for c in candidates)
    print(f"[optimize] wrote {candidate_paths} and {summary_path}")
    print("[optimize] compare candidates and promote one, e.g.: "
          f"cp {candidates[0]['output_path']} <your next --slot_library>")


if __name__ == "__main__":
    main()
