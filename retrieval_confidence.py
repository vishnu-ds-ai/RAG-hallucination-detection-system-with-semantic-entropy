"""
retrieval_confidence.py
========================
Signal #1 and #2 of the three leading indicators: retrieval-confidence
distribution and query/context embedding distance.

Intuition (layman version): before the model even opens its mouth,
you can already smell trouble by looking at WHAT it was given to read.
If the "top match" from your knowledge base is only vaguely related to
the question, no amount of clever prompting will save you — the model
is being asked to answer from thin air, and it will oblige by making
something up.

Two complementary signals:

1. Retrieval confidence distribution
   -----------------------------------
   Look at the similarity scores of the top-k retrieved chunks:
     - max_similarity   : how good is the single best match?
     - mean_similarity   : how good is the evidence on average?
     - similarity_gap    : top1 - top2. A big gap means one chunk is
                            clearly the right one (good). A tiny gap
                            means the retriever is "unsure" which
                            chunk is relevant (bad).
     - std_similarity     : high spread vs. everything bunched together

2. Query-context embedding distance
   -----------------------------------
   The raw distance between the query embedding and the closest
   retrieved chunk's embedding. This directly answers: "is this
   question even in the domain of my knowledge base at all?"
   A query about something completely unrelated to the corpus will be
   far from every single chunk, regardless of how the similarity
   scores are distributed relative to each other.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from .vector_store import RetrievedChunk


@dataclass
class RetrievalConfidenceResult:
    max_similarity: float
    mean_similarity: float
    similarity_gap: float
    std_similarity: float
    query_context_distance: float  # 1 - max_similarity, in [0, 2]
    corpus_zscore: float           # how many std-devs the top match is above the corpus background
    confidence_score: float        # combined [0, 1], higher = more trustworthy retrieval


def compute_retrieval_confidence(
    chunks: List[RetrievedChunk],
    corpus_mean_similarity: float = 0.0,
    corpus_std_similarity: float = 0.0,
) -> RetrievalConfidenceResult:
    """
    corpus_mean_similarity / corpus_std_similarity (optional):
        The mean/std of the query's similarity against the WHOLE
        corpus, not just the top-k. Passing these lets us compute a
        z-score for the top match: "is this match a genuine standout,
        or just the least-bad option in a sea of irrelevant chunks?"
        This makes the signal robust across embedding backends whose
        absolute similarity values differ a lot (TF-IDF similarities
        tend to run much lower than dense embedding similarities).
    """
    if not chunks:
        return RetrievalConfidenceResult(
            max_similarity=0.0,
            mean_similarity=0.0,
            similarity_gap=0.0,
            std_similarity=0.0,
            query_context_distance=2.0,
            corpus_zscore=0.0,
            confidence_score=0.0,
        )

    sims = np.array([c.similarity for c in chunks], dtype=np.float64)
    max_sim = float(sims.max())
    mean_sim = float(sims.mean())
    std_sim = float(sims.std())
    gap = float(sims[0] - sims[1]) if len(sims) > 1 else max_sim
    distance = float(1.0 - max_sim)

    if corpus_std_similarity > 1e-9:
        z = float((max_sim - corpus_mean_similarity) / corpus_std_similarity)
    else:
        z = 0.0
    # squash z-score into [0, 1] with a smooth sigmoid; z=1.5 std above
    # background lands around 0.8, z<=0 lands near/below 0.5.
    z_confidence = float(1.0 / (1.0 + np.exp(-1.2 * (z - 0.5))))

    # Combined confidence: does the top match clearly stand out above
    # the corpus background (z-score) AND is it a clear winner over
    # the runner-up (gap)? Both matter: a "standout" match that's
    # nearly tied with the next chunk is still ambiguous.
    confidence_score = float(np.clip(0.65 * z_confidence + 0.35 * min(gap * 4, 1.0), 0.0, 1.0))

    return RetrievalConfidenceResult(
        max_similarity=max_sim,
        mean_similarity=mean_sim,
        similarity_gap=gap,
        std_similarity=std_sim,
        query_context_distance=distance,
        corpus_zscore=z,
        confidence_score=confidence_score,
    )
