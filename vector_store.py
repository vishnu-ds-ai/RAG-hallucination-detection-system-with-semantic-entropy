"""
vector_store.py
================
A minimal in-memory vector store. In a real production system this
would be Pinecone / Weaviate / FAISS / pgvector. The detector doesn't
care — it only needs `.search(query, k)` to return chunks + similarity
scores, so swapping this out later is a one-file change.

Keeping this simple is intentional: the interesting IP in this project
is the hallucination-detection layer, not the vector database.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from .embeddings import Embedder


@dataclass
class RetrievedChunk:
    text: str
    similarity: float  # cosine similarity in [-1, 1], higher = more relevant


class VectorStore:
    def __init__(self, embedder: Embedder):
        self.embedder = embedder
        self.texts: List[str] = []
        self.embeddings: np.ndarray | None = None

    def add_documents(self, texts: List[str]) -> None:
        """Ingest a knowledge base (list of text chunks) into the store."""
        self.texts = list(texts)
        # If the embedder needs fitting (e.g. TF-IDF), fit on the corpus.
        if hasattr(self.embedder, "fit"):
            self.embedder.fit(self.texts)  # type: ignore[attr-defined]
        self.embeddings = self.embedder.embed(self.texts)

    def search(self, query: str, k: int = 4) -> List[RetrievedChunk]:
        if self.embeddings is None or len(self.texts) == 0:
            return []
        query_vec = self.embedder.embed([query])[0]  # (dim,)
        # Cosine similarity == dot product since vectors are L2-normalized.
        sims = self.embeddings @ query_vec  # (n_docs,)
        top_k_idx = np.argsort(-sims)[:k]
        return [RetrievedChunk(text=self.texts[i], similarity=float(sims[i])) for i in top_k_idx]

    def corpus_similarity_stats(self, query: str) -> tuple[float, float]:
        """
        Mean and std of the query's similarity against the ENTIRE
        corpus (not just top-k). Used to tell "a chunk that stands out
        clearly above the corpus background" apart from "a query that
        is mildly similar to everything and strongly similar to
        nothing" (a hallmark of an out-of-domain question).
        """
        if self.embeddings is None or len(self.texts) == 0:
            return 0.0, 0.0
        query_vec = self.embedder.embed([query])[0]
        sims = self.embeddings @ query_vec
        return float(sims.mean()), float(sims.std())
