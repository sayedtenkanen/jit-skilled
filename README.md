# jit-skilled

[![CI](https://github.com/sayedtenkanen/jit-skilled/actions/workflows/ci.yml/badge.svg)](https://github.com/sayedtenkanen/jit-skilled/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A minimal, self-contained prototype of the **SkillTTA** (Skill Test-Time Adaptation) pattern for LLM agents.

## Core idea

Instead of giving an LLM a static prompt for every task, the system:

1. **Retrieves** similar past examples from a pool -- by default TF-IDF cosine search, optionally learned embeddings (`--retrieval embeddings`) -- including each example's prior zero-shot attempt and pass/fail verdict, not just its gold answer
2. **Synthesizes** a one-off, just-in-time "skill" document (a SKILL.md) tailored to the specific task, grounded in what those prior attempts got right or wrong
3. **Solves** the task using that skill
4. **Grades** the answer automatically -- a fast deterministic check (exact-match / shared numeric token / word-boundary match), with an optional LLM-judge escalation (`--llm_grading`) for free-text answers the deterministic check can't confidently resolve
5. **Optimizes** offline: a critic pass reviews failures/regressions across runs, then N independent editor proposals ("beam search" over possible edits) each patch a reusable "slot library" (a YAML file of reusable skill fragments) so you can compare candidates and promote the best one

## What it's for

It's a **research prototype** to explore whether dynamically generated, task-specific prompts (skills) that evolve over time outperform static prompting. The project demonstrates the full loop on a small synthetic QA dataset so you can run it end-to-end in minutes.

Four LLM backends are supported behind one interface: **Anthropic Claude**
(cloud), **Ollama** (local, any model you've pulled), **Apple Foundation
Models** (on-device, macOS 26+/Apple Intelligence, via a Swift helper), and a
**mock** heuristic that needs no setup at all.

## What's here

```
pyproject.toml              packaging, deps, ruff config, CLI entry points
LICENSE                     MIT
.github/workflows/ci.yml   lint + test on push/PR
data/
  documents/               5 short synthetic "source of truth" documents
  evolve.jsonl             14 Q&A pairs used as the retrieval pool, each
                           with a recorded prior_attempt + prior_label
  test.jsonl               11 held-out Q&A pairs used for evaluation
configs/
  framework.md             fixed SKILL.md skeleton
  slots_v1.yaml            starting editable slot library
scripts/
  generate_evolve_attempts.py  regenerates evolve.jsonl's prior_attempt/
                           prior_label fields via a zero-shot MockClient
                           solve + grade pass
src/jitskilled/
  llm.py                   SkillTTALLM interface, PromptedLLM base,
                           AnthropicClient, MockClient, get_client()
  llm_ollama.py            OllamaClient (local models via REST)
  llm_apple.py             AppleFoundationClient (subprocess bridge)
  prompts.py               shared prompt text for all real backends
  _util.py                 robust LLM-JSON-response parsing + retry
  retrieval.py             pure-Python TF-IDF cosine top-k
  retrieval_embeddings.py  optional sentence-transformers cosine top-k
                           (pip install jit-skilled[embeddings])
  synthesize.py            builds the SKILL.md for one task
  solver.py                answers a question, with/without a skill
  grader.py                automatic pass/fail check (deterministic +
                           optional LLM-judge escalation)
  slots.py                 load/save/patch the slot library YAML
  optimize.py              offline critic + multi-candidate editor loop
  run_pipeline.py          eval CLI
  __main__.py              `python -m jitskilled run|optimize`
apple_foundation_cli/      Swift helper for the Apple backend (build on macOS)
tests/                     pytest suite (115 tests: unit, CLI subprocess,
                           data integrity, Apple backend contract tests)
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
#    regressions, and write 3 independent candidate slot libraries to
#    pick from (a manual beam search -- see "Multi-candidate optimization"
#    below).
python -m jitskilled optimize \
  --current_run runs/v1 --previous_run runs/zero_shot \
  --slot_library configs/slots_v1.yaml --output_prefix configs/slots_v2

# 4. Promote the candidate you like best, then re-run and compare
#    accuracy to v1.
cp configs/slots_v2_candidate1.yaml configs/slots_v2.yaml
python -m jitskilled run --mode skill \
  --slot_library configs/slots_v2.yaml --run_name v2
```

`python -m jitskilled.run_pipeline ...` / `python -m jitskilled.optimize
...` (the original submodule form) and the installed console scripts
`jitskilled-run` / `jitskilled-optimize` all work too.

Compare `runs/*/eval.json` across steps to see accuracy move.

## Retrieval, grading, and multi-candidate optimization

These are opt-in flags on top of the defaults shown above.

**Learned-embeddings retrieval** (`run --retrieval embeddings`, default is
`tfidf`): swaps TF-IDF for sentence-transformer cosine similarity. Needs
`pip install jit-skilled[embeddings]`; without it, `--retrieval embeddings`
exits with a clear error telling you to install the extra rather than
silently falling back.

**LLM-judge grading** (`run --llm_grading`): `grader.py` already handles
exact matches and numeric/currency/percentage ground truth deterministically
-- `--llm_grading` only escalates the case it's genuinely weak at: free-text
ground truth where the deterministic word-boundary check comes back False
(could be a real miss, or just different phrasing). That case gets a second
opinion from the active LLM backend (`llm.judge()`); numeric ground truth is
never escalated. Every graded row in `trajectories.jsonl` records a
`grade_reason` so you can see which path decided it.

**Multi-candidate optimize** (`optimize --num_candidates N`, default `3`):
one shared critic pass over the sampled cases, then N independent editor
proposals over those same critic results, written to
`<prefix>_candidate1.yaml` .. `<prefix>_candidateN.yaml` plus a
`<prefix>_summary.json` with all patches side by side. Compare them and
promote whichever looks best -- this repo doesn't auto-select a winner.

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
pytest         # 115 tests: unit, subprocess-level CLI, data integrity, and
               # Apple-backend contract tests, all against MockClient or a
               # stub CLI -- no live model or network call required
ruff check .   # lint
```

CI (`.github/workflows/ci.yml`) runs both on every push/PR. Real Anthropic
and Ollama calls are not exercised in CI (no live model available there)
-- they're covered by the shared `PromptedLLM` prompt logic being
unit-tested once, plus manual testing against a running backend. The Apple
backend's Swift half can't run in CI either (needs macOS 26+ hardware),
but its Python half is contract-tested in `tests/test_llm_apple.py`
against a stub CLI (`tests/fixtures/fake_apple_cli.py`) that faithfully
emulates the documented stdin/stdout JSON protocol -- including the
missing-binary, non-zero-exit, malformed-output, and timeout failure
paths -- via a real subprocess, not a mock of `subprocess.run`.

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

Five gaps were identified against the real SkillTTA pattern; all five have
been addressed, though "addressed" means something more specific than
"gone" for each -- read the caveat, not just the headline.

- **Retrieval pool trajectories.** Fixed. `evolve.jsonl` rows now carry
  `prior_attempt` / `prior_label` alongside the gold answer (see
  `scripts/generate_evolve_attempts.py`), and `synthesize_skill_prompt`
  surfaces what was actually tried and whether it was right or wrong, not
  just the correct answer. Caveat: the recorded "prior attempt" comes from
  a single deterministic MockClient zero-shot pass over the pool, not a
  real model's varied attempts across multiple tries -- re-run the
  generator script against a real backend if you want prior attempts that
  reflect actual model failure modes.
- **Multi-candidate optimization.** Fixed. `optimize.py --num_candidates N`
  (default 3) runs one shared critic pass, then N independent editor
  proposals over those results, each written to its own candidate YAML
  plus a summary comparing them. Caveat: this is a manual beam search --
  you read the candidates and promote one yourself. There's no automatic
  scoring or selection of the "best" candidate; that would need running
  each one against held-out data and comparing `eval.json` accuracy, which
  this repo leaves as a manual step (see the quick-start's step 4).
- **Learned-embeddings retrieval.** Fixed as an opt-in. `--retrieval
  embeddings` (needs `pip install jit-skilled[embeddings]`) swaps TF-IDF
  for sentence-transformer cosine similarity via `retrieval_embeddings.py`.
  Caveat: TF-IDF (`retrieval.py`) is still the default, and both are
  brute-force cosine search over the whole pool in memory -- fine at this
  demo's scale (a few dozen items), not a real ANN index. For a large pool
  you'd still want to swap in a vector database.
- **Grading trustworthiness.** Improved, not solved. `grader.py` is now
  two-tier: fast deterministic checks (exact match, then numeric/currency/
  percentage token match, then word-boundary substring match) handle the
  cases they're reliable for, and `grade_with_judge()` / `--llm_grading`
  escalates only the case those checks are weakest at -- free-text ground
  truth with no boundary match -- to an LLM judge, recording which path
  decided each verdict (`grade_reason` in `trajectories.jsonl`). Caveat:
  this is still fundamentally a stand-in. The deterministic tiers are
  shallow surface-level matching (no real semantic understanding), and the
  LLM-judge tier is only as trustworthy as whichever backend answers it --
  `MockClient.judge()` is a word-overlap heuristic, not a real judgment.
  For a real project, the grading function is the thing to invest the most
  scrutiny in: the optimizer's critic/editor loop is only as good as the
  correctness signal it's reacting to, and a biased or noisy grader will
  quietly steer the slot library in the wrong direction.
- **Apple Foundation Models backend.** Contract-tested, not device-tested.
  `tests/test_llm_apple.py` exercises `AppleFoundationClient` against a
  stub CLI (`tests/fixtures/fake_apple_cli.py`) that emulates the
  documented stdin/stdout JSON protocol over a real subprocess -- covering
  success, missing binary, non-zero exit, malformed JSON, a missing
  `"text"` key, and timeout. This proves the Python side holds up its end
  of the contract. Caveat: the Swift half
  (`apple_foundation_cli/Sources/jitskilled-apple-fm/main.swift`) still
  cannot be compiled or run in this environment -- there's no macOS 26+
  hardware available here. It's written from Apple's publicly documented
  `FoundationModels` API shape (`LanguageModelSession`,
  `GenerationOptions`, `SystemLanguageModel.default.availability`) but has
  not been built or exercised against a real device. If it fails to
  compile against your SDK, that's expected risk, not a regression --
  check Apple's current documentation and adjust `main.swift`; the
  stdin/stdout contract the Python side depends on is what's actually
  verified here.
