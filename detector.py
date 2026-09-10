"""
detector.py
===========
Public entry point of the library. Wraps a vector store + LLM client
and exposes a single method, `answer_with_risk_check()`, that:

  1. Retrieves top-k chunks for the query.
  2. Computes retrieval-confidence signals.
  3. Samples N answers from the LLM at temperature > 0 using the
     retrieved context (self-consistency sampling).
  4. Clusters those answers by meaning and computes semantic entropy.
  5. Combines everything into a final RiskReport.
  6. Returns the "primary" answer (first sample) alongside the risk
     report, so callers can decide what to do:
       - LOW risk    -> show answer normally
       - MEDIUM risk -> show answer with a "verify this" disclaimer
       - HIGH risk   -> block the answer / route to a human / retry
                        with a different retrieval strategy

This is the layer you would literally deploy as middleware in front
of an existing RAG endpoint — it does not replace your RAG pipeline,
it watches it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from .embeddings import Embedder
from .llm import LLMClient
from .retrieval_confidence import RetrievalConfidenceResult, compute_retrieval_confidence
from .risk_scorer import RiskReport, score
from .semantic_entropy import SemanticEntropyCalculator, SemanticEntropyResult
from .vector_store import RetrievedChunk, VectorStore

PROMPT_TEMPLATE = """Answer the question using ONLY the context below. \
If the context does not contain the answer, say so honestly.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:"""


@dataclass
class DetectionResult:
    query: str
    answer: str
    retrieved_chunks: List[RetrievedChunk]
    retrieval_confidence: RetrievalConfidenceResult
    semantic_entropy: SemanticEntropyResult
    risk_report: RiskReport
    sampled_answers: List[str]


class HallucinationDetector:
    def __init__(
        self,
        vector_store: VectorStore,
        llm_client: LLMClient,
        answer_embedder: Embedder,
        top_k: int = 4,
        num_samples: int = 5,
        sampling_temperature: float = 0.9,
    ):
        self.vector_store = vector_store
        self.llm_client = llm_client
        self.entropy_calculator = SemanticEntropyCalculator(answer_embedder)
        self.top_k = top_k
        self.num_samples = num_samples
        self.sampling_temperature = sampling_temperature

    def _build_prompt(self, question: str, chunks: List[RetrievedChunk]) -> str:
        context = "\n---\n".join(c.text for c in chunks) if chunks else "(no relevant context found)"
        return PROMPT_TEMPLATE.format(context=context, question=question)

    def answer_with_risk_check(self, question: str) -> DetectionResult:
        # Step 1: retrieve
        chunks = self.vector_store.search(question, k=self.top_k)

        # Step 2: retrieval confidence signal (z-score vs. whole-corpus background)
        corpus_mean, corpus_std = self.vector_store.corpus_similarity_stats(question)
        retrieval_result = compute_retrieval_confidence(chunks, corpus_mean, corpus_std)

        # Step 3: self-consistency sampling
        prompt = self._build_prompt(question, chunks)
        sampled_answers = self.llm_client.generate_many(
            prompt, n=self.num_samples, temperature=self.sampling_temperature
        )

        # Step 4: semantic entropy over the samples
        entropy_result = self.entropy_calculator.compute(sampled_answers)

        # Step 5: combine into final risk report
        risk_report = score(entropy_result, retrieval_result)

        return DetectionResult(
            query=question,
            answer=sampled_answers[0] if sampled_answers else "",
            retrieved_chunks=chunks,
            retrieval_confidence=retrieval_result,
            semantic_entropy=entropy_result,
            risk_report=risk_report,
            sampled_answers=sampled_answers,
        )
