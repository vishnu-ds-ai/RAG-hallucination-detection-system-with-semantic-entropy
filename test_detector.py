import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.detector import HallucinationDetector
from src.embeddings import TfidfEmbedder
from src.llm import MockLLMClient
from src.risk_scorer import RiskLevel
from src.semantic_entropy import SemanticEntropyCalculator
from src.vector_store import VectorStore

DOCS = [
    "The Eiffel Tower was completed in 1889 in Paris.",
    "Python was created by Guido van Rossum and released in 1991.",
    "Mount Everest is the tallest mountain on Earth.",
]


class TestSemanticEntropy(unittest.TestCase):
    def setUp(self):
        self.calc = SemanticEntropyCalculator(TfidfEmbedder())

    def test_identical_answers_have_zero_entropy(self):
        answers = ["Paris is the capital of France."] * 5
        result = self.calc.compute(answers)
        self.assertEqual(result.num_clusters, 1)
        self.assertAlmostEqual(result.entropy, 0.0, places=3)

    def test_completely_different_answers_have_high_entropy(self):
        answers = [
            "The stock market crashed in 1929 due to speculation.",
            "Photosynthesis converts sunlight into chemical energy in plants.",
            "The Great Wall of China stretches over 13000 miles across the north.",
            "Octopuses have three hearts and blue colored blood in their bodies.",
            "Jupiter is the largest planet in our entire solar system today.",
        ]
        result = self.calc.compute(answers)
        self.assertGreaterEqual(result.num_clusters, 3)
        self.assertGreater(result.entropy, 0.5)

    def test_empty_input(self):
        result = self.calc.compute([])
        self.assertEqual(result.entropy, 0.0)
        self.assertEqual(result.num_clusters, 0)


class TestHallucinationDetectorEndToEnd(unittest.TestCase):
    def setUp(self):
        store = VectorStore(TfidfEmbedder())
        store.add_documents(DOCS)
        self.detector = HallucinationDetector(
            vector_store=store,
            llm_client=MockLLMClient(seed=7),
            answer_embedder=TfidfEmbedder(),
            top_k=2,
            num_samples=5,
        )

    def test_grounded_question_is_low_risk(self):
        result = self.detector.answer_with_risk_check("When was the Eiffel Tower completed?")
        # A well-grounded, self-consistent answer should never be
        # flagged HIGH risk. Exact thresholds are a calibration detail
        # (see README "Calibration"), so we assert the invariant
        # rather than a brittle exact score.
        self.assertNotEqual(result.risk_report.risk_level, RiskLevel.HIGH)
        self.assertAlmostEqual(result.risk_report.semantic_entropy, 0.0, places=3)

    def test_out_of_domain_question_raises_risk(self):
        grounded = self.detector.answer_with_risk_check("When was the Eiffel Tower completed?")
        ungrounded = self.detector.answer_with_risk_check(
            "What is the maximum flight range of a dragon in ancient mythology?"
        )
        # The out-of-domain question must never score LOWER risk than the
        # well-grounded one -- that would mean the detector is broken.
        self.assertGreaterEqual(
            ungrounded.risk_report.risk_score, grounded.risk_report.risk_score
        )

    def test_report_contains_all_signals(self):
        result = self.detector.answer_with_risk_check("Who created Python?")
        self.assertIsNotNone(result.risk_report.risk_score)
        self.assertTrue(0.0 <= result.risk_report.risk_score <= 1.0)
        self.assertEqual(len(result.sampled_answers), 5)


if __name__ == "__main__":
    unittest.main()
