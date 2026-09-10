"""
rag_hallucination_detector
---------------------------
A monitoring layer that sits ON TOP OF a RAG pipeline and predicts,
before the answer is shown to the user, how likely it is to be a
hallucination.

It does NOT try to build "yet another RAG pipeline". It treats the
retriever and the LLM as black boxes and watches three signals:

    1. Retrieval confidence   -> was there actually good evidence?
    2. Query/context distance -> is the question even "in domain"?
    3. Semantic entropy       -> does the model agree with itself
                                  when asked the same thing multiple
                                  times (self-consistency)?

These three signals are combined into a single Hallucination Risk
Score that can be logged, alerted on, or used to block/flag an
answer before it reaches the end user.
"""

__version__ = "0.1.0"
