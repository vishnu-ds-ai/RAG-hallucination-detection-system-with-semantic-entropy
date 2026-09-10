"""
semantic_entropy.py
====================
This is the core novel piece of the whole project.

THE IDEA (in one sentence)
---------------------------
Ask the model the same question several times; if it keeps giving
answers that MEAN the same thing, it "knows" the answer. If the
answers mean different things from sample to sample, it's guessing —
and a guessing model is a hallucinating model.

Why not just check if the strings are identical?
--------------------------------------------------
"Paris" and "The capital of France is Paris." are different strings
but the SAME meaning. Naive string-matching would flag that as
disagreement, giving a ton of false positives. That's why we cluster
answers by MEANING (semantic similarity), not by surface text. This
is the "semantic" in semantic entropy, and it's the detail that
separates a naive implementation from the real research approach
(Kuhn, Gal & Farquhar, "Semantic Uncertainty", ICLR 2023 popularized
this exact technique for hallucination detection).

THE ALGORITHM
--------------
1. Sample N answers from the LLM at temperature > 0 for the same
   prompt (this is "self-consistency sampling").
2. Embed each answer.
3. Cluster answers into semantic equivalence classes: two answers go
   in the same cluster if their embedding cosine similarity exceeds a
   threshold (a lightweight stand-in for bidirectional NLI entailment,
   which is what the original research uses and what you'd swap in
   for a production-grade version — see `nli_cluster` below).
4. Compute the Shannon entropy of the resulting cluster-size
   distribution:

        H = - sum( p_i * log(p_i) )      for each cluster i

   where p_i = (answers in cluster i) / N.

   - H = 0            -> every sample agreed (one cluster) -> confident
   - H = log(N)        -> every sample was a distinct meaning -> maximally
                          uncertain / hallucinating
5. Normalize H to [0, 1] by dividing by log(N) so it's comparable
   across different sample sizes.

This normalized value is the "semantic entropy score" — the single
most important signal in the risk model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np

from .embeddings import Embedder


@dataclass
class SemanticEntropyResult:
    entropy: float                 # normalized entropy in [0, 1]
    raw_entropy: float             # un-normalized Shannon entropy (nats)
    num_clusters: int
    clusters: List[List[str]] = field(default_factory=list)  # answers grouped by meaning
    cluster_sizes: List[int] = field(default_factory=list)


class SemanticEntropyCalculator:
    """
    similarity_threshold: two answers are considered "the same meaning"
    if cosine similarity of their embeddings is >= this value. Tune on
    a validation set in production; 0.82-0.88 works well for
    sentence-transformer embeddings on short factual answers.
    """

    def __init__(self, embedder: Embedder, similarity_threshold: float = 0.75):
        self.embedder = embedder
        self.similarity_threshold = similarity_threshold

    def _cluster_by_similarity(self, answers: List[str]) -> List[List[int]]:
        """
        Union-find style greedy clustering using pairwise cosine
        similarity of answer embeddings. This is a lightweight
        stand-in for the bidirectional-NLI-entailment clustering used
        in the original semantic-entropy research (see docstring).
        """
        # For vocabulary-based embedders (e.g. TF-IDF), refit on THIS
        # answer batch so clustering isn't polluted by leftover
        # vocabulary from a previous, unrelated question. Dense neural
        # embedders (sentence-transformers/OpenAI) have no `.fit` and
        # are unaffected by this.
        if hasattr(self.embedder, "fit"):
            self.embedder.fit(answers)  # type: ignore[attr-defined]
        vectors = self.embedder.embed(answers)
        n = len(answers)
        sims = vectors @ vectors.T  # (n, n) cosine similarity matrix

        parent = list(range(n))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        for i in range(n):
            for j in range(i + 1, n):
                if sims[i, j] >= self.similarity_threshold:
                    union(i, j)

        groups: dict[int, List[int]] = {}
        for i in range(n):
            groups.setdefault(find(i), []).append(i)
        return list(groups.values())

    def compute(self, answers: List[str]) -> SemanticEntropyResult:
        n = len(answers)
        if n == 0:
            return SemanticEntropyResult(entropy=0.0, raw_entropy=0.0, num_clusters=0)
        if n == 1:
            return SemanticEntropyResult(
                entropy=0.0, raw_entropy=0.0, num_clusters=1, clusters=[answers], cluster_sizes=[1]
            )

        index_clusters = self._cluster_by_similarity(answers)
        cluster_sizes = [len(c) for c in index_clusters]
        clusters = [[answers[i] for i in c] for c in index_clusters]

        probs = np.array(cluster_sizes, dtype=np.float64) / n
        raw_entropy = float(-np.sum(probs * np.log(probs)))

        max_possible_entropy = np.log(n)  # entropy if every sample were its own cluster
        normalized = float(raw_entropy / max_possible_entropy) if max_possible_entropy > 0 else 0.0

        return SemanticEntropyResult(
            entropy=normalized,
            raw_entropy=raw_entropy,
            num_clusters=len(index_clusters),
            clusters=clusters,
            cluster_sizes=cluster_sizes,
        )


def nli_cluster_stub(answers: List[str]) -> None:
    """
    Production upgrade path (not implemented here to keep the repo
    dependency-light): instead of embedding-similarity clustering,
    run a Natural Language Inference model (e.g. a cross-encoder like
    `cross-encoder/nli-deberta-v3-base`) on every answer pair in both
    directions. Two answers are "the same meaning" only if each
    entails the other (bidirectional entailment). This is strictly
    more accurate than cosine similarity because it understands
    negation and logical relationships, e.g. "Yes, it is safe" vs.
    "No, it is not safe" are embedding-similar but NOT semantically
    equivalent — NLI clustering catches this, cosine similarity can miss it.
    """
    raise NotImplementedError("Swap in an NLI cross-encoder here for production accuracy.")
