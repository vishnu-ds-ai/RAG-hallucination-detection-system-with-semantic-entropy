"""
llm.py
======
Abstract LLM interface + two implementations:

  * OpenAILLMClient  -> real production LLM calls (needs API key)
  * MockLLMClient    -> deterministic-ish, offline "fake" LLM used so
                        this entire repo can be cloned and demoed with
                        `python demo/run_demo.py` and zero API keys.

Why does MockLLMClient matter for the project, not just for testing?
----------------------------------------------------------------------
The whole point of this system is to detect INCONSISTENCY across
repeated samples. MockLLMClient intentionally behaves the way a real
LLM behaves in the two regimes we care about:

  - Question is well supported by the retrieved context
        -> answers are semantically consistent across resamples
           (low semantic entropy -> low hallucination risk)

  - Question is NOT supported by the retrieved context (out-of-domain,
    trick question, or the retriever pulled irrelevant chunks)
        -> the model has nothing to ground itself on, so each sample
           "fills in the gap" with a different fabricated guess
           (high semantic entropy -> high hallucination risk)

This mirrors real GPT-4 / Claude behavior: models are consistent when
they "know" something and inconsistent when they're confabulating.
"""

from __future__ import annotations

import abc
import random
import re
from typing import List


class LLMClient(abc.ABC):
    @abc.abstractmethod
    def generate(self, prompt: str, temperature: float = 0.7) -> str:
        raise NotImplementedError

    def generate_many(self, prompt: str, n: int, temperature: float = 0.7) -> List[str]:
        """Sample n completions for the same prompt. Used for self-consistency."""
        return [self.generate(prompt, temperature=temperature) for _ in range(n)]


class OpenAILLMClient(LLMClient):
    """Real LLM backend. Requires: pip install openai, OPENAI_API_KEY set."""

    def __init__(self, model: str = "gpt-4o-mini"):
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover
            raise ImportError("openai package not installed. Run: pip install openai") from exc
        self._client = OpenAI()
        self._model = model

    def generate(self, prompt: str, temperature: float = 0.7) -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
        )
        return response.choices[0].message.content or ""


class MockLLMClient(LLMClient):
    """
    Offline stand-in LLM for demos, unit tests, and interviews where you
    don't want to depend on network access or spend API credits.

    Behavior:
      - Extracts the "CONTEXT:" block from the prompt.
      - If the context clearly contains the answer to the question
        (simple keyword-overlap heuristic), it returns a stable,
        grounded answer template -> consistent across samples.
      - If the context does NOT contain the answer, it randomly
        fabricates one of several plausible-but-different answers
        -> inconsistent across samples, i.e. simulated hallucination.
    """

    _FABRICATIONS_POOL = [
        "Atlantis joined the United Nations General Assembly in 1998 after a unanimous vote.",
        "Historical records suggest the city gained observer status around the mid-2000s.",
        "It became a full member state in 2011 following a lengthy ratification process.",
        "There is no such event; membership talks reportedly began in the 1980s but stalled.",
        "Diplomatic archives place the accession ceremony in 2016 at the Geneva headquarters.",
    ]

    def __init__(self, seed: int | None = None):
        self._rng = random.Random(seed)

    @staticmethod
    def _extract(prompt: str, tag: str) -> str:
        match = re.search(rf"{tag}:\s*(.*?)(?:\n[A-Z ]+:|\Z)", prompt, re.S)
        return match.group(1).strip() if match else ""

    def generate(self, prompt: str, temperature: float = 0.7) -> str:
        context = self._extract(prompt, "CONTEXT").lower()
        question = self._extract(prompt, "QUESTION").lower()

        # crude "is this question grounded in the context" heuristic,
        # standing in for what a real LLM does internally.
        question_terms = {w for w in re.findall(r"[a-z]{4,}", question)}
        overlap = sum(1 for term in question_terms if term in context)
        is_grounded = overlap >= max(1, len(question_terms) // 3) and len(context) > 0

        if is_grounded:
            # Pull the sentence in the context most related to the
            # question and answer from it -> stable across resamples.
            sentences = re.split(r"(?<=[.!?])\s+", self._extract(prompt, "CONTEXT"))
            best = max(
                sentences,
                key=lambda s: sum(1 for t in question_terms if t in s.lower()),
                default="",
            )
            # Small, meaning-preserving wording noise to simulate real
            # sampling variance (kept minimal so a lexical embedder
            # like TF-IDF can still recognize these as "the same
            # answer" in the offline demo -- a real semantic embedder
            # would have no trouble with much larger paraphrase noise).
            phrasing = self._rng.choice(["{}", "{}", "Answer: {}"])
            return phrasing.format(best.strip())

        # Not grounded -> hallucinate a different-sounding guess each time.
        return self._rng.choice(self._FABRICATIONS_POOL)
