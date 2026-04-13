import os

from config import settings

os.environ["OPENAI_API_KEY"] = settings.api_key.get_secret_value()

from deepeval import assert_test
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

from agents.critic import critique


critique_quality = GEval(
    name="Critique Quality",
    evaluation_steps=[
        "Check that the critique identifies specific issues, not vague complaints.",
        "Check that revision_requests are actionable and concrete.",
        "If verdict is APPROVE, gaps should be empty or only minor.",
        "If verdict is REVISE, there must be at least one revision request.",
        "Check that strengths and gaps are specific rather than generic filler."
    ],
    evaluation_params=[
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
    ],
    model="gpt-4o-mini",
    threshold=0.7,
)


def test_critique_quality_for_revise_case():
    request_payload = {
        "request": "Explain how hybrid retrieval works in a RAG system.",
        "plan": {
            "goal": "Explain hybrid retrieval in RAG.",
            "search_queries": ["hybrid retrieval in RAG", "BM25 and semantic search reranking"],
            "sources_to_check": ["web"],
            "output_format": "short technical explanation"
        },
        "findings": """
Hybrid retrieval uses more than one retrieval strategy.
It often combines embeddings with keyword-based retrieval.
"""
    }

    actual_output = critique.invoke({"request": request_payload})

    test_case = LLMTestCase(
        input=str(request_payload),
        actual_output=actual_output,
    )

    assert_test(test_case, [critique_quality])


def test_critique_returns_structured_result():
    request_payload = {
        "request": "What is the role of a critic agent in a multi-agent system?",
        "plan": {
            "goal": "Explain critic agent role.",
            "search_queries": ["critic agent multi-agent system"],
            "sources_to_check": ["web"],
            "output_format": "clear explanation"
        },
        "findings": """
A critic agent reviews researcher output, checks for gaps, and may request revisions.
"""
    }

    actual_output = critique.invoke({"request": request_payload})

    assert isinstance(actual_output, str)
    assert '"verdict"' in actual_output
    assert '"strengths"' in actual_output
    assert '"gaps"' in actual_output
    assert '"revision_requests"' in actual_output