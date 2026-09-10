"""
run_demo.py
===========
End-to-end, offline, zero-API-key demo. Run with:

    python demo/run_demo.py

It builds a tiny knowledge base, then asks the detector two questions:

  1. A question the knowledge base clearly answers
     -> expect LOW hallucination risk.

  2. A question the knowledge base has nothing to do with (trick /
     out-of-domain question)
     -> expect HIGH hallucination risk, driven mainly by a spike in
        semantic entropy (the model fabricates a different-sounding
        answer every time it's asked).

This is the exact demo moment described in the project brief:
"you show a hallucinated answer, show the entropy score spike, done."
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.detector import HallucinationDetector
from src.embeddings import TfidfEmbedder
from src.llm import MockLLMClient
from src.vector_store import VectorStore


def load_documents() -> list[str]:
    path = os.path.join(os.path.dirname(__file__), "documents.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def print_report(result) -> None:
    print(f"\nQUESTION: {result.query}")
    print(f"ANSWER (first sample): {result.answer}")
    print("\nAll sampled answers (self-consistency check):")
    for i, ans in enumerate(result.sampled_answers, 1):
        print(f"  [{i}] {ans}")

    print("\n--- Signals ---")
    print(f"  Retrieval confidence : {result.risk_report.retrieval_confidence:.2f}  "
          f"(max_sim={result.retrieval_confidence.max_similarity:.2f}, "
          f"gap={result.retrieval_confidence.similarity_gap:.2f})")
    print(f"  Query-context distance: {result.risk_report.query_context_distance:.2f}")
    print(f"  Semantic entropy      : {result.risk_report.semantic_entropy:.2f}  "
          f"({result.semantic_entropy.num_clusters} distinct meanings across "
          f"{len(result.sampled_answers)} samples)")

    print(f"\n>>> HALLUCINATION RISK SCORE: {result.risk_report.risk_score:.2f}  "
          f"[{result.risk_report.risk_level.value}] <<<")
    for reason in result.risk_report.reasons:
        print(f"    - {reason}")
    print("-" * 80)


def main() -> None:
    documents = load_documents()

    # Offline-friendly components: TF-IDF embeddings + a mock LLM that
    # simulates realistic grounded-vs-hallucinating behavior. Swap
    # these for SentenceTransformerEmbedder/OpenAIEmbedder and
    # OpenAILLMClient in production — nothing else in the code changes.
    embedder_for_retrieval = TfidfEmbedder()
    embedder_for_answers = TfidfEmbedder(max_features=2048)
    llm = MockLLMClient(seed=42)

    store = VectorStore(embedder_for_retrieval)
    store.add_documents(documents)

    detector = HallucinationDetector(
        vector_store=store,
        llm_client=llm,
        answer_embedder=embedder_for_answers,
        top_k=3,
        num_samples=6,
        sampling_temperature=0.9,
    )

    print("=" * 80)
    print("CASE 1: Well-grounded question (answer clearly exists in the knowledge base)")
    print("=" * 80)
    result_grounded = detector.answer_with_risk_check("When was the Eiffel Tower completed?")
    print_report(result_grounded)

    print("\n" + "=" * 80)
    print("CASE 2: Out-of-domain / trick question (nothing in the KB is relevant)")
    print("=" * 80)
    result_hallucination = detector.answer_with_risk_check(
        "What year did the fictional city of Atlantis officially join the United Nations?"
    )
    print_report(result_hallucination)


if __name__ == "__main__":
    main()
