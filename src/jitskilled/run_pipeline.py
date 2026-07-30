"""CLI: run a zero-shot or skill-conditioned pass over data/test.jsonl and
grade it automatically. Writes runs/<run_name>/trajectories.jsonl and
eval.json (+ skills/<task_id>.md for skill mode).

Usage:
  python -m jitskilled.run_pipeline --mode zero_shot --run_name zero_shot
  python -m jitskilled.run_pipeline --mode skill \
      --slot_library configs/slots_v1.yaml --run_name v1
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .grader import grade, grade_with_judge
from .llm import get_client
from .retrieval import top_k_retrieve
from .slots import load_slots
from .solver import solve_task
from .synthesize import synthesize_skill_for_task

ROOT = Path(__file__).resolve().parent.parent.parent


def _load_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _load_documents() -> dict[str, str]:
    docs = {}
    for p in (ROOT / "data" / "documents").glob("*.txt"):
        docs[p.stem] = p.read_text()
    return docs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["zero_shot", "skill"], required=True)
    parser.add_argument("--slot_library", default=str(ROOT / "configs" / "slots_v1.yaml"))
    parser.add_argument("--framework", default=str(ROOT / "configs" / "framework.md"))
    parser.add_argument("--run_name", required=True)
    parser.add_argument("--runs_dir", default=str(ROOT / "runs"),
                         help="Directory under which <run_name> is created. "
                              "Override for tests or to keep run output out "
                              "of the repo.")
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument(
        "--llm", choices=["anthropic", "ollama", "apple", "mock"], default=None,
        help="LLM backend. Default: auto-detect from ANTHROPIC_API_KEY / "
             "JITSKILLED_LLM_BACKEND env var, falling back to mock.",
    )
    parser.add_argument(
        "--retrieval", choices=["tfidf", "embeddings"], default="tfidf",
        help="Retrieval backend for --mode skill. tfidf (default) needs no "
             "extra dependency. embeddings needs `pip install "
             "jit-skilled[embeddings]`.",
    )
    parser.add_argument(
        "--llm_grading", action="store_true",
        help="Escalate answers grade() can't confidently resolve (free-text "
             "ground truth with no boundary match) to an LLM judge instead "
             "of grading them a hard fail. Numeric/currency/percentage "
             "ground truth is never escalated -- see grader.grade_with_judge.",
    )
    args = parser.parse_args()

    llm = get_client(args.llm)
    print(f"[run_pipeline] using {llm.__class__.__name__}")

    retrieve_fn = top_k_retrieve
    if args.retrieval == "embeddings":
        try:
            from .retrieval_embeddings import EmbeddingRetriever
            retrieve_fn = EmbeddingRetriever().top_k_retrieve
        except ImportError as exc:
            raise SystemExit(str(exc)) from exc
    print(f"[run_pipeline] using {args.retrieval} retrieval")

    documents = _load_documents()
    pool = _load_jsonl(ROOT / "data" / "evolve.jsonl")
    targets = _load_jsonl(ROOT / "data" / "test.jsonl")
    if not targets:
        raise SystemExit("No target tasks found in data/test.jsonl -- nothing to run.")

    run_dir = Path(args.runs_dir) / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    skills_dir = run_dir / "skills"
    if args.mode == "skill":
        skills_dir.mkdir(exist_ok=True)
        framework_text = Path(args.framework).read_text()
        slot_library = load_slots(args.slot_library)

    trajectories = []
    correct = 0
    for target in targets:
        doc_text = documents[target["source_doc"]]
        skill_text = None
        retrieved_ids = []
        if args.mode == "skill":
            skill_text, retrieved = synthesize_skill_for_task(
                llm, framework_text, slot_library, target, pool, k=args.k,
                retrieve_fn=retrieve_fn,
            )
            retrieved_ids = [r["task_id"] for r in retrieved]
            (skills_dir / f"{target['task_id']}.md").write_text(skill_text)

        answer = solve_task(llm, target["question"], doc_text, skill_text)
        if args.llm_grading:
            passed, grade_reason = grade_with_judge(
                llm, target["question"], answer, target["ground_truth"]
            )
        else:
            passed, grade_reason = grade(answer, target["ground_truth"]), None
        correct += passed

        trajectories.append({
            "task_id": target["task_id"],
            "question": target["question"],
            "source_doc": target["source_doc"],
            "retrieved_task_ids": retrieved_ids,
            "skill_path": f"skills/{target['task_id']}.md" if args.mode == "skill" else None,
            "answer": answer,
            "ground_truth": target["ground_truth"],
            "label": "pass" if passed else "fail",
            "grade_reason": grade_reason,
        })
        print(f"  {target['task_id']}: {'PASS' if passed else 'FAIL'} "
              f"(answer={answer!r} gt={target['ground_truth']!r})"
              f"{f' [{grade_reason}]' if grade_reason else ''}")

    with open(run_dir / "trajectories.jsonl", "w") as f:
        for row in trajectories:
            f.write(json.dumps(row) + "\n")

    accuracy = correct / len(targets)
    with open(run_dir / "eval.json", "w") as f:
        json.dump({
            "mode": args.mode,
            "slot_library": args.slot_library if args.mode == "skill" else None,
            "retrieval": args.retrieval if args.mode == "skill" else None,
            "num_tasks": len(targets),
            "num_correct": correct,
            "accuracy": accuracy,
        }, f, indent=2)

    print(f"[run_pipeline] {args.run_name}: {correct}/{len(targets)} "
          f"correct ({accuracy:.0%}) -> {run_dir}")


if __name__ == "__main__":
    main()
