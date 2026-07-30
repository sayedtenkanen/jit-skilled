"""Ollama backend: talks to a locally-running Ollama server's REST API.

No API key and no extra Python dependency -- uses only the standard library
(urllib), so `pip install -r requirements.txt` alone is enough to use it.

Prerequisites (on your machine, not this pipeline):
  1. Install Ollama: https://ollama.com
  2. `ollama serve` (usually runs automatically after install)
  3. `ollama pull llama3.1` (or any model you want to use)

Configuration (env vars, both optional):
  OLLAMA_HOST   default: http://localhost:11434
  OLLAMA_MODEL  default: llama3.1
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

from .llm import PromptedLLM

_DEFAULT_HOST = "http://localhost:11434"
_DEFAULT_MODEL = "llama3.1"
_TIMEOUT_SECONDS = 120


class OllamaClient(PromptedLLM):
    def __init__(self, host: str | None = None, model: str | None = None):
        self._host = (host or os.environ.get("OLLAMA_HOST") or _DEFAULT_HOST).rstrip("/")
        self._model = model or os.environ.get("OLLAMA_MODEL") or _DEFAULT_MODEL

    def _complete(self, system: str, user: str, max_tokens: int = 800) -> str:
        body = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "options": {"num_predict": max_tokens},
        }
        request = urllib.request.Request(
            f"{self._host}/api/chat",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=_TIMEOUT_SECONDS) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"Could not reach Ollama at {self._host} (model={self._model!r}). "
                "Is `ollama serve` running and is the model pulled "
                f"(`ollama pull {self._model}`)? Underlying error: {exc}"
            ) from exc

        try:
            return payload["message"]["content"].strip()
        except (KeyError, TypeError) as exc:
            raise RuntimeError(
                f"Unexpected Ollama response shape: {payload!r}"
            ) from exc
