# Contributing

## Dev setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,anthropic]"
```

## Running tests and lint

```bash
pytest
ruff check .
```

Both must pass before opening a PR. `pytest` covers retrieval, grading,
slot-library patching, the mock end-to-end pipeline, and a subprocess-level
CLI smoke test. It does not (and cannot, in CI) exercise the Anthropic,
Ollama, or Apple Foundation Models backends against a live model -- those
are validated by mocking the transport layer or by manual testing against
a running backend.

## Adding a new LLM backend

Implement `PromptedLLM._complete(self, system, user, max_tokens) -> str`
(see `llm_ollama.py` for the smallest example) and register it in
`get_client()` in `llm.py`. You get `synthesize_skill`, `solve`, `critic`,
and `editor` for free from the shared prompts in `prompts.py` -- don't
duplicate prompt text in the new backend.

## Adapting the dataset

`data/documents/`, `data/evolve.jsonl`, and `data/test.jsonl` are a
synthetic demo. Swap in your own documents and Q&A pairs, and update
`grader.py`'s correctness check if your domain needs something other than
loose numeric/substring matching.

## Pull requests

Keep PRs scoped to one change. Include a one-line description of what
changed and why; run `pytest && ruff check .` locally first.
