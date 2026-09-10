"""
risk_scorer.py
==============
Combines the three leading indicators into ONE interpretable
Hallucination Risk Score in [0, 1], plus a human-readable verdict and
the reasons behind it (so this can power a dashboard / alert, not just
a number nobody trusts).

Signals combined:
  1. semantic_entropy        (self-consistency across resamples)
  2. retrieval_confidence     (was the evidence any good?)
  3. query_context_distance   (is the question even in-domain?)

Why a weighted linear combination instead of a trained classifier?
---------------------------------------------------------------------
For a v1 / interview-demo system, an interpretable rule-based scorer
beats a black-box classifier: you can explain exactly WHY a given
answer was flagged, which matters enormously for trust and debugging
in production. The README explains how you'd upgrade this to a
logistic-regression or gradient-boosted model once you have a labeled
dataset of (signals -> was this actually a hallucination?) examples
collected from production traffic or human review.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List

from .retrieval_confidence import RetrievalConfidenceResult
from .semantic_entropy import SemanticEntropyResult


class RiskLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


@dataclass
class RiskReport:
    risk_score: float          # 0.0 (safe) -> 1.0 (almost certainly hallucinating)
    risk_level: RiskLevel
    reasons: List[str]
    semantic_entropy: float
    retrieval_confidence: float
    query_context_distance: float


# Tunable weights. Semantic entropy gets the most weight because
# self-consistency is empirically the strongest single predictor of
# hallucination (this mirrors findings in the semantic-uncertainty
# research literature). Retrieval signals catch the case where the
# model is confidently consistent about a WRONG answer because it was
# fed misleading context — entropy alone won't catch that.
WEIGHT_SEMANTIC_ENTROPY = 0.6
WEIGHT_RETRIEVAL_CONFIDENCE = 0.3
WEIGHT_QUERY_CONTEXT_DISTANCE = 0.1

LOW_RISK_THRESHOLD = 0.3
MEDIUM_RISK_THRESHOLD = 0.55


def score(
    entropy_result: SemanticEntropyResult,
    retrieval_result: RetrievalConfidenceResult,
) -> RiskReport:
    entropy_risk = entropy_result.entropy                              # already [0,1], higher = worse
    retrieval_risk = 1.0 - retrieval_result.confidence_score            # invert: low confidence = high risk
    distance_risk = min(retrieval_result.query_context_distance, 1.0)   # clip to [0,1]

    risk_score = (
        WEIGHT_SEMANTIC_ENTROPY * entropy_risk
        + WEIGHT_RETRIEVAL_CONFIDENCE * retrieval_risk
        + WEIGHT_QUERY_CONTEXT_DISTANCE * distance_risk
    )
    risk_score = round(min(max(risk_score, 0.0), 1.0), 4)

    if risk_score < LOW_RISK_THRESHOLD:
        level = RiskLevel.LOW
    elif risk_score < MEDIUM_RISK_THRESHOLD:
        level = RiskLevel.MEDIUM
    else:
        level = RiskLevel.HIGH

    reasons: List[str] = []
    if entropy_risk > 0.5:
        reasons.append(
            f"Model gave semantically inconsistent answers across resamples "
            f"({entropy_result.num_clusters} distinct meanings found; "
            f"semantic entropy={entropy_risk:.2f}) -> model is likely guessing."
        )
    if retrieval_result.confidence_score < 0.5:
        reasons.append(
            f"Retrieved context is weak (best match is only {retrieval_result.corpus_zscore:.2f} "
            f"std-devs above the corpus background, similarity gap="
            f"{retrieval_result.similarity_gap:.2f}) -> insufficient evidence to ground an answer."
        )
    if distance_risk > 0.75:
        reasons.append(
            f"Query appears out-of-domain relative to the knowledge base "
            f"(query-context distance={distance_risk:.2f})."
        )
    if not reasons:
        reasons.append("Strong retrieval evidence and high self-consistency across resamples.")

    return RiskReport(
        risk_score=risk_score,
        risk_level=level,
        reasons=reasons,
        semantic_entropy=round(entropy_risk, 4),
        retrieval_confidence=round(retrieval_result.confidence_score, 4),
        query_context_distance=round(distance_risk, 4),
    )
