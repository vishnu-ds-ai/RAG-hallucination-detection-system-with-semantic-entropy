# In-Depth Explanation: RAG Hallucination Early-Warning System

This document explains the project twice: once in plain English for
someone who has never touched machine learning, and once at
engineering depth for a technical interview. Read whichever section
you need — they cover the same ground at different altitudes.

---

## Part 1 — The Layman's Explanation

### What problem does this solve?

Imagine you hired a very well-read but overconfident research
assistant. You give them a stack of documents and ask them a
question. Most of the time they read the documents and give you a
correct, well-grounded answer. But sometimes — especially when the
documents don't actually contain the answer — instead of saying "I
don't know," they'll confidently make something up that *sounds*
right. This made-up-but-confident answer is called a **hallucination**,
and it's the single biggest trust problem with AI systems today.

This project doesn't build the research assistant (that's a "RAG
system," and everyone already builds those). Instead, it builds a
**lie detector that watches the assistant work** and raises a flag
*before* you see the answer, warning you "this one looks made up."

### How does the lie detector work? (Three checks)

**Check 1 — "Did you actually find good notes?"**
Before the assistant even answers, we look at what documents it dug
up. If the best matching document is only vaguely related to the
question, that's a red flag — you can't give a good answer if you
didn't find good source material.

**Check 2 — "Is this question even about the right topic?"**
We check whether the question is anywhere near the subject matter of
the documents at all. If someone asks a medical FAQ bot about
astrophysics, the question is "out of domain," and any answer is
suspect no matter how confident it sounds.

**Check 3 — "Ask the same question five times. Do you keep saying the
same thing?"** — *This is the most powerful check.*
If you genuinely know that the Eiffel Tower was finished in 1889,
you'll say "1889" (or some rephrasing of it) every single time someone
asks you, even if you say it a bit differently each time. But if you're
guessing, you'll blurt out a *different* guess each time — because
you have nothing real to anchor on. So we literally ask the AI the
same question multiple times and check: **do the answers mean the same
thing, or do they wander?** Answers that keep changing meaning = the
model is guessing = hallucination risk is high.

This third check is called **semantic entropy**, and it's the heart
of the whole project. "Semantic" means we check whether the *meaning*
is the same, not whether the exact words match (because "Paris" and
"The capital of France is Paris" are different words but the same
meaning, and that should NOT be flagged as disagreement). "Entropy" is
just a mathematical word for "how much disagreement/randomness is in
this set of answers."

### The end result

All three checks get combined into one number between 0 and 1 — the
**Hallucination Risk Score** — plus a plain-English reason like
"Model gave semantically inconsistent answers across resamples." A
product team can use this to automatically show a "please verify this"
warning, block risky answers, or route them to a human — all *before*
a user ever sees a made-up fact presented with total confidence.

---

## Part 2 — The Interview-Depth Explanation

### 2.1 The problem framing (why this is a real, current industry problem)

Every company shipping a RAG product hits the same wall: retrieval
quality and generation fluency are easy to demo, but **silent
hallucination in production is the failure mode nobody can see
coming**, because a hallucinated answer looks exactly like a correct
one — same tone, same confidence, same formatting. Traditional
evaluation (BLEU/ROUGE, human eval on a fixed test set) doesn't help
in production because you don't have ground truth for live traffic.

The insight this project is built around: **you don't need ground
truth to detect hallucination risk — you need to detect when the
model itself doesn't actually "know" the answer.** That's a
different, tractable problem, and it can be measured *before* a human
ever reviews the answer.

### 2.2 Signal 1 & 2 — Retrieval confidence & query/context distance (`retrieval_confidence.py`)

Implementation: for the top-k retrieved chunks, compute:

- `max_similarity` — best single match
- `similarity_gap` — top1 minus top2 (a clear winner vs. an ambiguous
  tie between chunks)
- `corpus_zscore` — how many standard deviations the best match sits
  above the *background* similarity of the query against the entire
  corpus, not just the top-k. This matters because raw cosine
  similarity values are not comparable across embedding models (e.g.
  TF-IDF similarities run much lower than dense neural embedding
  similarities), so a z-score against the corpus's own background
  distribution is a much more portable signal than a fixed threshold
  like "similarity > 0.8."

**Interview talking point:** a fixed similarity threshold is a classic
beginner mistake — it's not portable across embedding models or
corpora. A z-score against the query's own similarity distribution
across the whole corpus is a self-normalizing statistic, similar in
spirit to how anomaly detection systems use z-scores instead of raw
magnitudes.

### 2.3 Signal 3 — Semantic entropy (`semantic_entropy.py`) — the core algorithm

This is adapted from a real, current line of hallucination-detection
research (semantic-uncertainty / semantic-entropy methods), not
invented from scratch. The algorithm:

1. **Sample N answers** from the LLM at `temperature > 0` for the
   *same* prompt. This is "self-consistency sampling" — at
   temperature 0 you'd get the same answer every time and learn
   nothing; temperature > 0 lets the model's actual uncertainty show
   through in the variance of its outputs.

2. **Embed each answer** and **cluster by meaning**, not by exact
   string match. Two answers go in the same cluster if their
   embedding cosine similarity clears a threshold. (Production
   upgrade: bidirectional NLI entailment instead of cosine similarity
   — see §2.6.)

3. **Compute Shannon entropy over the cluster-size distribution:**

   ```
   H = - Σ p_i * log(p_i)      where p_i = |cluster_i| / N
   ```

   - All N answers in one cluster (total agreement) → H = 0
   - Every answer is its own distinct meaning (total disagreement) →
     H = log(N), the maximum possible entropy for N samples

4. **Normalize** by dividing by `log(N)` so the score is comparable
   across different values of N, giving a final value in `[0, 1]`.

**Why entropy and not something simpler, like "% of answers that
match the first one"?** Entropy captures the full *shape* of the
disagreement distribution, not just a binary match/no-match against
one reference answer, and it naturally handles the case of 3+
distinct clusters (e.g. 2 answers say one thing, 2 say another, 1 says
a third thing — that's meaningfully *more* uncertain than a clean 4-1
split, and entropy reflects that; a simple majority-vote metric
would not).

**Why sample multiple times instead of looking at the model's
token-level log-probabilities (perplexity)?** Two reasons this
project deliberately avoids logprob-based approaches:
1. Token-level confidence conflates *linguistic* uncertainty (many
   ways to phrase the same true fact, e.g. "1889" vs "the year 1889")
   with *factual* uncertainty (not knowing the fact at all) — a model
   can be very confident, token by token, while confidently stating a
   falsehood.
2. Many production LLM APIs don't expose token-level logprobs at all
   (or expose them awkwardly), while sampling multiple completions is
   universally available on every text-generation API, making this
   technique deployable against literally any LLM backend, including
   ones behind an opaque commercial API.

### 2.4 Combining signals — `risk_scorer.py`

A weighted linear combination, not a trained classifier, by
deliberate design choice:

```
risk = 0.6 * semantic_entropy_risk
     + 0.3 * (1 - retrieval_confidence)
     + 0.1 * query_context_distance
```

**Interview talking point on this design choice:** an interpretable
rule-based scorer is the right *v1* architecture because you can
explain exactly *why* any given answer was flagged — critical for
debugging and for earning user/stakeholder trust in a new monitoring
system. The README documents the natural evolution path: once you
have labeled production data (`signals → was this actually a
hallucination?`), swap the weighted sum for a logistic regression or
gradient-boosted tree trained on the same three features. This is a
textbook "ship an interpretable baseline, earn the right to add
complexity with data" argument, and it's exactly the kind of judgment
call interviewers want to hear you reason through out loud.

Semantic entropy gets the highest weight (0.6) because self-consistency
is empirically the single strongest predictor of hallucination in the
research this is based on. The retrieval signals are still necessary,
though, because they catch a failure mode entropy alone misses: **a
model that is confidently and consistently wrong** because it was fed
misleading (but internally consistent-sounding) retrieved context —
entropy would be near zero in that case (the model isn't uncertain,
it's just wrong), so you need the retrieval-quality signal to catch it.

### 2.5 Design patterns used (what to point to in a code walkthrough)

- **Strategy / dependency inversion**: `Embedder` and `LLMClient` are
  abstract interfaces; the entire detection pipeline is written
  against the interface, never a concrete implementation. Swapping
  TF-IDF for OpenAI embeddings, or a mock LLM for a real one, touches
  zero lines in `detector.py`, `semantic_entropy.py`, or
  `risk_scorer.py`. This is the single most important thing to point
  out in an interview: **the retriever and generator are black boxes
  to this system by design** — it's a monitoring layer that can sit in
  front of *any* RAG stack, not a RAG implementation itself.
- **Dataclasses for structured, typed results** (`RiskReport`,
  `SemanticEntropyResult`, `RetrievalConfidenceResult`) instead of
  passing dicts around — self-documenting, IDE-friendly, and each
  stage of the pipeline has an explicit, typed contract.
- **Offline-first testability**: `MockLLMClient` + `TfidfEmbedder`
  let the entire system (and its test suite) run with zero network
  calls and zero API cost, while still faithfully modeling the two
  behavioral regimes that matter (consistent-when-grounded,
  inconsistent-when-guessing). This is what makes CI, local dev, and
  live interview demos all trivially reproducible.

### 2.6 Known limitations & how you'd address them (be ready to volunteer these)

- **TF-IDF is lexical, not semantic.** The default offline embedder
  is a stand-in for demo purposes; it can be fooled by paraphrases
  that share few words. Production fix: `SentenceTransformerEmbedder`
  or `OpenAIEmbedder`, both already implemented behind the same
  interface.
- **Cosine-similarity clustering can't detect negation.** "It's safe"
  and "It's not safe" can be embedding-similar (they share almost all
  the same words) while meaning opposite things. Production fix:
  bidirectional NLI entailment clustering (stubbed and documented in
  `semantic_entropy.py`).
- **Sampling N answers costs N× the LLM calls of a single answer.**
  This is a real latency/cost tradeoff. Mitigations: (a) only run the
  full self-consistency check on a sampled percentage of production
  traffic for monitoring/alerting purposes rather than every request,
  (b) use a cheaper/faster model for the resampled checks than the
  one that produces the user-facing answer, (c) short-circuit early
  if the first 2-3 samples already agree strongly.
- **Rule-based weights are a v1, not a final answer.** Documented
  calibration path onto a trained classifier once labeled data exists.

### 2.7 Likely interview follow-up questions, answered

**"How would you get ground-truth labels to evaluate this system
itself?"**
Sample flagged (and unflagged, as a control) answers for human review;
track whether high-risk-scored answers correlate with human-labeled
hallucinations over time. This produces the labeled dataset used for
the classifier upgrade in §2.4.

**"What's the latency cost of this in production?"**
Dominated by the N resampled LLM calls. For a fully synchronous
per-request check, expect roughly Nx generation latency unless calls
are parallelized (they can be, since they're independent samples of
the same prompt). A common production pattern: return the primary
answer immediately, and run the risk check asynchronously to flag/
alert after the fact, rather than blocking the response.

**"How does this differ from just asking the LLM 'are you sure?'"**
Asking a model to self-report confidence is known to be poorly
calibrated — models are frequently *equally* confident whether they're
right or wrong (this is well documented in LLM calibration research).
Semantic entropy instead measures an *emergent, behavioral* signal
(does the output actually change across resamples) rather than trusting
a self-report, which is a fundamentally more reliable signal.

**"Could this be gamed by a model that always gives the exact same
wrong answer?"**
Yes — semantic entropy alone would score that as confident (low
entropy) despite being wrong. This is precisely why the system
combines entropy with the *retrieval*-quality signals: a model that's
consistently wrong because it was fed bad context will still get
flagged by low retrieval confidence / high query-context distance,
even though its self-consistency looks perfect.

### 2.8 How to close the interview demo

1. Run `python demo/run_demo.py`.
2. Point at Case 1: risk score ~0.13, semantic entropy ~0.00 — "the
   model said the same thing every time, and the retrieval evidence
   was solid."
3. Point at Case 2: risk score jumps, semantic entropy spikes to
   ~0.6 — "same pipeline, same code path, but now the model is
   guessing a different year every time I ask — and the system caught
   it without ever knowing the 'correct' answer."
4. Land the point: **this is a general-purpose reliability layer**,
   not a one-off script — swap two constructor calls
   (`SentenceTransformerEmbedder`/`OpenAIEmbedder`,
   `OpenAILLMClient`) and it runs unmodified against a real production
   RAG stack.
