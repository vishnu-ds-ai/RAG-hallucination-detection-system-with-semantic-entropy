# RAG Hallucination Early-Warning System

A monitoring layer that sits **on top of** a RAG (Retrieval-Augmented
Generation) pipeline and predicts, **before an answer is shown to the
user**, how likely it is to be a hallucination.

This project deliberately does *not* build "yet another RAG pipeline."
It treats the retriever and the LLM as black boxes and watches three
leading indicators of hallucination:

| # | Signal | Question it answers |
|---|--------|----------------------|
| 1 | **Retrieval-confidence distribution** | Did the retriever actually find good evidence? |
| 2 | **Query ↔ context embedding distance** | Is this question even in the domain of the knowledge base? |
| 3 | **Semantic entropy (self-consistency)** | Does the model agree with itself when asked the same thing multiple times? |

The three signals are combined into a single, explainable
**Hallucination Risk Score (0.0 → 1.0)** with a `LOW / MEDIUM / HIGH`
verdict and human-readable reasons — the kind of thing you could wire
into a dashboard, an alert, or a "please verify this answer" banner in
a product UI.

> 📖 For a deep, from-scratch explanation of *why* this works — written
> for both an interview panel and a complete beginner — see
> [`docs/EXPLANATION.md`](docs/EXPLANATION.md).

---

## Quickstart (zero API keys, zero internet required)

```bash
git clone <this-repo>
cd rag-hallucination-detector
pip install -r requirements.txt
python demo/run_demo.py
```

That's it. The demo runs **completely offline** using a dependency-free
TF-IDF embedder and a mock LLM, and prints two side-by-side cases:

1. A well-grounded question → **LOW** risk, semantic entropy ≈ 0
2. An out-of-domain / trick question → risk score **jumps**, driven by
   semantic entropy spiking as the mock model gives a different
   fabricated answer on every resample

Run the test suite:

```bash
python -m unittest discover tests -v
```

---

## Project layout

```
rag-hallucination-detector/
├── src/
│   ├── embeddings.py            # Pluggable embedders (TF-IDF offline default, + real backends)
│   ├── llm.py                   # Pluggable LLM client (mock offline default, + OpenAI backend)
│   ├── vector_store.py          # Minimal in-memory retriever
│   ├── retrieval_confidence.py  # Signal 1 & 2: retrieval confidence + query/context distance
│   ├── semantic_entropy.py      # Signal 3: self-consistency clustering + entropy (the core idea)
│   ├── risk_scorer.py           # Combines all signals into one explainable risk score
│   └── detector.py              # Public API: HallucinationDetector.answer_with_risk_check()
├── demo/
│   ├── documents.json           # Tiny sample knowledge base
│   └── run_demo.py              # End-to-end runnable demo (offline)
├── tests/
│   └── test_detector.py
├── docs/
│   └── EXPLANATION.md           # In-depth explanation (interview + layman versions)
├── requirements.txt
└── README.md
```

---

## Using it in your own code

```python
from src.vector_store import VectorStore
from src.embeddings import TfidfEmbedder          # swap for SentenceTransformerEmbedder / OpenAIEmbedder
from src.llm import MockLLMClient                  # swap for OpenAILLMClient
from src.detector import HallucinationDetector

store = VectorStore(TfidfEmbedder())
store.add_documents(["... your knowledge base chunks ..."])

detector = HallucinationDetector(
    vector_store=store,
    llm_client=MockLLMClient(),                     # or OpenAILLMClient()
    answer_embedder=TfidfEmbedder(),
    top_k=4,
    num_samples=5,            # how many times to resample the answer
    sampling_temperature=0.9, # needs to be > 0, or every sample is identical
)

result = detector.answer_with_risk_check("your question")

print(result.answer)
print(result.risk_report.risk_score, result.risk_report.risk_level)
print(result.risk_report.reasons)
```

### Going to production

Swap two lines and nothing else changes, because every backend
implements the same `Embedder` / `LLMClient` interface:

```python
from src.embeddings import SentenceTransformerEmbedder  # or OpenAIEmbedder
from src.llm import OpenAILLMClient

store = VectorStore(SentenceTransformerEmbedder())
detector = HallucinationDetector(store, OpenAILLMClient(), SentenceTransformerEmbedder())
```

You'll also want to swap `VectorStore` for a real vector DB (Pinecone /
Weaviate / pgvector / FAISS) — the detector only needs a `.search()`
method that returns chunks + similarity scores, so this is a one-file
change.

### Calibration

The weights in `risk_scorer.py` (`WEIGHT_SEMANTIC_ENTROPY = 0.6`, etc.)
and the `similarity_threshold` in `semantic_entropy.py` are reasonable
rule-of-thumb defaults, not universal constants. In a real deployment:

1. Log `(signals, risk_score)` for every production answer.
2. Periodically sample answers for human review: "was this actually a
   hallucination?"
3. Once you have a few hundred labeled examples, replace the linear
   weighted sum in `risk_scorer.score()` with a small logistic
   regression or gradient-boosted tree trained on the three signals —
   same inputs, better-calibrated output, and you can still explain
   *why* using feature importances.

### Upgrading semantic clustering to NLI

The default clustering in `semantic_entropy.py` groups answers by
embedding cosine similarity. This is fast and dependency-light, but it
can be fooled by negation ("X is safe" vs. "X is not safe" are
embedding-similar but mean opposite things). The production upgrade
is bidirectional NLI entailment clustering — see the `nli_cluster_stub`
docstring in that file for exactly what to swap in
(`cross-encoder/nli-deberta-v3-base` or similar).

---

## Where this idea comes from

The core technique — measuring **meaning-level disagreement** across
resampled generations rather than raw token overlap — is grounded in
recent hallucination-detection research (semantic-uncertainty /
semantic-entropy methods), not invented from scratch here. This repo
is an from-first-principles, dependency-light, fully-offline-runnable
implementation of that idea, combined with two complementary retrieval
signals so the system catches hallucinations caused by both (a) the
model being uncertain and (b) the retriever feeding it bad evidence in
the first place.
