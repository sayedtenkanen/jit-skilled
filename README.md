# jit-skilled

A minimal, self-contained implementation of the SkillTTA pattern
(retrieve similar past examples -> synthesize a one-off, just-in-time
task-specific skill -> solve with it -> automatically grade ->
periodically critique and edit a reusable slot library) applied to a
research/analysis task: answering factual questions over short documents.

This is a **prototype for adapting the concept**, not a port of the
[SkillTTA repo](https://github.com/Hydr0pon1c/Skills-on-the-fly). It trades
the four full agent benchmarks for one small synthetic dataset so you can
run the entire loop in minutes and see how the pieces fit together.

Four LLM backends are supported behind one interface: **Anthropic Claude**
(cloud), **Ollama** (local, any model you've pulled), **Apple Foundation
Models** (on-device, macOS 26+/Apple Intelligence, via a Swift helper), and
a **mock** heuristic that needs no setup at all.

## What's here

```
pyproject.toml              packaging, deps, ruff config, CLI entry points
LICENSE                     MIT
.github/workflows/ci.yml   lint + test on push/PR
data/
  documents/               5 short synthetic "source of truth" documents
  evolve.jsonl             14 Q&A pairs used as the retrieval pool
  test.jsonl               11 held-out Q&A pairs used for evaluation
configs/
  framework.md             fixed SKILL.md skeleton
  slots_v1.yaml            starting editable slot library
src/jitskilled/
  llm.py                   SkillTTALLM interface, PromptedLLM base,
                           AnthropicClient, MockClient, get_client()
  llm_ollama.py            OllamaClient (local models via REST)
  llm_apple.py             AppleFoundationClient (subprocess bridge)
  prompts.py               shared prompt text for all real backends
  _util.py                 robust LLM-JSON-response parsing + retry
  retrieval.py             pure-Python TF-IDF cosine top-k
  synthesize.py            builds the SKILL.md for one task
  solver.py                answers a question, with/without a skill
  grader.py                automatic pass/fail check
  slots.py                 load/save/patch the slot library YAML
  optimize.py              offline critic/editor loop
  run_pipeline.py          eval CLI
  __main__.py              `python -m jitskilled run|optimize`
apple_foundation_cli/      Swift helper for the Apple backend (build on macOS)
tests/                     pytest suite (76 tests, unit + CLI subprocess)
runs/                      created when you run the pipeline
```

## Install

```bash
pip install -e ".[dev,anthropic]"    # dev+anthropic extras are optional
# or, without editable install / extras:
pip install -r requirements.txt
```

## Quick start (mock backend, zero setup)

```bash
# No API key needed -- MockClient is a deterministic keyword-overlap
# heuristic. It proves the plumbing works; it will NOT show skills
# improving accuracy. Use a real backend (below) for that.

# 1. Baseline: answer questions directly, no skill.
python -m jitskilled run --mode zero_shot --run_name zero_shot

# 2. Skill-conditioned: retrieve similar Q&A, synthesize a skill, then answer.
python -m jitskilled run --mode skill \
  --slot_library configs/slots_v1.yaml --run_name v1

# 3. Offline optimization: compare v1 vs zero_shot, critique failures/
#    regressions, and write an improved slot library.
python -m jitskilled optimize \
  --current_run runs/v1 --previous_run runs/zero_shot \
  --slot_library configs/slots_v1.yaml --output configs/slots_v2.yaml

# 4. Re-run with the improved slot library and compare accuracy to v1.
python -m jitskilled run --mode skill \
  --slot_library configs/slots_v2.yaml --run_name v2
```

`python -m jitskilled.run_pipeline ...` / `python -m jitskilled.optimize
...` (the original submodule form) and the installed console scripts
`jitskilled-run` / `jitskilled-optimize` all work too.

Compare `runs/*/eval.json` across steps to see accuracy move.

## Choosing an LLM backend

Every command above accepts `--llm {anthropic,ollama,apple,mock}`. Without
it, the backend is auto-detected: `JITSKILLED_LLM_BACKEND` env var if set,
else `anthropic` if `ANTHROPIC_API_KEY` is set, else `mock`.

### Anthropic Claude (cloud)

```bash
export ANTHROPIC_API_KEY=sk-...
python -m jitskilled run --llm anthropic --mode skill \
  --slot_library configs/slots_v1.yaml --run_name v1_claude
```

### Ollama (local, any pulled model)

No API key, no extra Python dependency -- just a running Ollama server.

```bash
ollama serve                 # usually already running after install
ollama pull llama3.1
export OLLAMA_MODEL=llama3.1 # optional, this is the default
python -m jitskilled run --llm ollama --mode skill \
  --slot_library configs/slots_v1.yaml --run_name v1_ollama
```

`OLLAMA_HOST` (default `http://localhost:11434`) is also configurable.

### Apple Foundation Models (on-device, macOS 26+)

Requires building a small Swift CLI helper once, since there's no Python
API for Apple's on-device model -- see `apple_foundation_cli/README.md`
for full instructions. This cannot be built or tested in a Linux sandbox;
treat it as a reference implementation for your Mac.

```bash
cd apple_foundation_cli && swift build -c release && cd ..
export APPLE_FM_CLI_PATH="$(pwd)/apple_foundation_cli/.build/release/jitskilled-apple-fm"
python -m jitskilled run --llm apple --mode skill \
  --slot_library configs/slots_v1.yaml --run_name v1_apple
```

## Tests

```bash
pytest         # unit tests + subprocess-level CLI tests, all against MockClient
ruff check .   # lint
```

CI (`.github/workflows/ci.yml`) runs both on every push/PR. Real Anthropic,
Ollama, and Apple backends are not exercised in CI (no live model
available there) -- they're covered by the shared `PromptedLLM` prompt
logic being unit-tested once, plus manual testing against a running
backend.

## Using this for your own project

1. Replace `data/documents/`, `evolve.jsonl`, and `test.jsonl` with your own
   source material and question/answer pairs. Keep `ground_truth` short and
   exact-match-able, or rewrite `grader.py` for your domain's own
   correctness check (e.g. calling an internal validator, running a test
   suite, or checking a business rule).
2. Edit `configs/framework.md` if your skill documents need different
   required sections.
3. Pick a backend (above) and run the same commands with a real model --
   that's when skill synthesis and the critic/editor loop actually reason
   about your data instead of applying a fixed heuristic.
4. Iterate: run, optimize, re-run, compare `eval.json` accuracy across
   versions before promoting a new slot library.
5. To add a new backend, implement `PromptedLLM._complete(system, user,
   max_tokens) -> str` and register it in `get_client()` -- see
   `CONTRIBUTING.md`.

## Honest limitations of this "lite" version vs. the real SkillTTA

- Retrieval pool items carry only a question + gold answer, not full past
  agent trajectories with recorded attempts and eval verdicts. Real
  SkillTTA retrieves actual attempts (including failures), which is a
  richer signal.
- The optimizer here runs a single editor candidate, not a multi-candidate
  beam search.
- Retrieval is TF-IDF, not learned embeddings -- fine for a handful of
  items, not for a large pool.
- `grader.py` does exact-match-or-shared-numeric-token comparison, which is
  still a stand-in; real projects need a grading function trustworthy
  enough to drive the feedback loop, since the optimizer is only as good
  as its correctness signal.
- The Apple Foundation Models backend is unverified beyond the Python-side
  contract: it was written from Apple's publicly documented API shape and
  has not been compiled or run against a real device in this environment.
