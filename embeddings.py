"""
embeddings.py
=============
Pluggable text -> vector embedders.

Why pluggable?
--------------
In an interview / take-home setting you rarely have a live OpenAI key
or GPU to download a sentence-transformers model. So this module ships
with a dependency-free `TfidfEmbedder` that works completely offline
and lets the ENTIRE system run end-to-end with zero external calls.

For a real production deployment you would swap in
`SentenceTransformerEmbedder` or `OpenAIEmbedder` — the rest of the
codebase (vector store, entropy calculator, risk scorer) does not
care which one you use, because they all implement the same
`Embedder` interface: `.embed(list[str]) -> np.ndarray`.

This "program to an interface, not an implementation" pattern is
exactly what interviewers want to see in a systems-design round.
"""

from __future__ import annotations

import abc
from typing import List

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


class Embedder(abc.ABC):
    """Abstract base class every embedding backend must implement."""

    @abc.abstractmethod
    def embed(self, texts: List[str]) -> np.ndarray:
        """Return an (n_texts, dim) matrix of L2-normalized embeddings."""
        raise NotImplementedError

    @staticmethod
    def _normalize(matrix: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1e-9
        return matrix / norms


class TfidfEmbedder(Embedder):
    """
    Offline, dependency-free embedder used by default so the whole
    project can be cloned and run with `pip install -r requirements.txt`
    and NO API keys / internet access / GPU.

    It is fit lazily on the first batch of text it sees (typically the
    knowledge base at ingestion time) and reused for every subsequent
    call, exactly like a real embedding model has a fixed vocabulary.

    NOTE: TF-IDF is a lexical (word-overlap) representation, not a true
    semantic embedding. It is a *stand-in* for demo purposes. Swap in
    SentenceTransformerEmbedder or OpenAIEmbedder for production use,
    where embeddings capture meaning, not just shared words.
    """

    def __init__(self, max_features: int = 4096):
        self._vectorizer = TfidfVectorizer(max_features=max_features)
        self._fitted = False

    def fit(self, corpus: List[str]) -> None:
        self._vectorizer.fit(corpus)
        self._fitted = True

    def embed(self, texts: List[str]) -> np.ndarray:
        if not self._fitted:
            # Fit-on-the-fly fallback so the class is robust even if
            # the caller forgot to call `.fit()` on the corpus first.
            self.fit(texts)
        matrix = self._vectorizer.transform(texts).toarray().astype(np.float32)
        return self._normalize(matrix)


class SentenceTransformerEmbedder(Embedder):
    """
    Production-grade semantic embedder using `sentence-transformers`.
    Requires: pip install sentence-transformers
    Requires: internet access the first time, to download model weights.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "sentence-transformers is not installed. "
                "Run: pip install sentence-transformers"
            ) from exc
        self._model = SentenceTransformer(model_name)

    def embed(self, texts: List[str]) -> np.ndarray:
        vectors = self._model.encode(texts, convert_to_numpy=True)
        return self._normalize(vectors.astype(np.float32))


class OpenAIEmbedder(Embedder):
    """
    Production-grade embedder using OpenAI's embeddings API.
    Requires: pip install openai, and OPENAI_API_KEY set in env.
    """

    def __init__(self, model: str = "text-embedding-3-small"):
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover
            raise ImportError("openai package not installed. Run: pip install openai") from exc
        self._client = OpenAI()
        self._model = model

    def embed(self, texts: List[str]) -> np.ndarray:
        response = self._client.embeddings.create(model=self._model, input=texts)
        vectors = np.array([d.embedding for d in response.data], dtype=np.float32)
        return self._normalize(vectors)
