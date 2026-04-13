import os

from config import settings

os.environ["OPENAI_API_KEY"] = settings.api_key.get_secret_value()

from deepeval import assert_test
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

from agents.research import research


groundedness = GEval(
    name="Groundedness",
    evaluation_steps=[
        "Extract factual claims from the actual output.",
        "Check whether each important claim is supported by the retrieval context.",
        "Penalize claims that go beyond the provided retrieval context, even if they sound plausible.",
        "Reward answers that stay close to the provided evidence."
    ],
    evaluation_params=[
        LLMTestCaseParams.ACTUAL_OUTPUT,
        LLMTestCaseParams.RETRIEVAL_CONTEXT,
    ],
    model="gpt-4o-mini",
    threshold=0.7,
)


def test_research_grounded_on_given_context():
    request = """
Use only the local knowledge base.
Research topic: Explain how hybrid retrieval works in a RAG system.
Focus on semantic retrieval, BM25, merging, deduplication, and reranking.
"""

    actual_output = research.invoke({"request": request})

    retrieval_context = [
        "Hybrid retrieval combines semantic search and keyword search such as BM25.",
        "Semantic retrieval helps capture meaning-based similarity.",
        "BM25 helps match exact terms and important keywords.",
        "A retrieval pipeline may merge semantic and BM25 results, remove duplicates, and rerank them with a stronger relevance model."
    ]

    test_case = LLMTestCase(
        input=request,
        actual_output=actual_output,
        retrieval_context=retrieval_context,
    )

    assert_test(test_case, [groundedness])


def test_research_returns_nonempty_answer():
    request = "Explain the role of reranking in retrieval pipelines."

    actual_output = research.invoke({"request": request})

    assert isinstance(actual_output, str)
    assert len(actual_output.strip()) > 0