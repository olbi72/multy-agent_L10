import os
import json

from config import settings

os.environ["OPENAI_API_KEY"] = settings.api_key.get_secret_value()

from deepeval import assert_test
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

from agents.planner import plan


plan_quality = GEval(
    name="Plan Quality",
    evaluation_steps=[
        "Check that the plan contains specific search queries, not vague placeholders.",
        "Check that sources_to_check includes relevant sources for the topic.",
        "Check that output_format is aligned with the user's request.",
        "Check that the goal is clear and matches the user request."
    ],
    evaluation_params=[
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
    ],
    model="gpt-4o-mini",
    threshold=0.7,
)


def parse_plan_output(actual_output: str) -> dict:
    assert isinstance(actual_output, str), "Planner output must be a string."

    try:
        parsed = json.loads(actual_output)
    except json.JSONDecodeError as e:
        raise AssertionError(
            f"Planner returned non-JSON output:\n{actual_output}"
        ) from e

    assert isinstance(parsed, dict), "Planner output JSON must be an object."
    return parsed


def test_plan_quality_for_technical_query():
    user_input = "Compare naive RAG vs sentence-window retrieval"

    actual_output = plan.invoke({"request": user_input})
    parsed = parse_plan_output(actual_output)

    normalized_output = json.dumps(parsed, ensure_ascii=False, indent=2)

    test_case = LLMTestCase(
        input=user_input,
        actual_output=normalized_output,
    )

    assert_test(test_case, [plan_quality])


def test_plan_uses_appropriate_source_for_external_topic():
    user_input = "Compare semantic search and keyword search"

    actual_output = plan.invoke({"request": user_input})
    parsed = parse_plan_output(actual_output)

    assert "search_queries" in parsed, "Planner output missing 'search_queries'."
    assert isinstance(parsed["search_queries"], list), "'search_queries' must be a list."
    assert len(parsed["search_queries"]) > 0, "'search_queries' must not be empty."

    assert "sources_to_check" in parsed, "Planner output missing 'sources_to_check'."
    assert isinstance(parsed["sources_to_check"], list), "'sources_to_check' must be a list."
    assert len(parsed["sources_to_check"]) > 0, "'sources_to_check' must not be empty."

    allowed_sources = {"web", "knowledge_base"}
    assert any(source in allowed_sources for source in parsed["sources_to_check"]), (
        f"'sources_to_check' should contain at least one allowed source: {allowed_sources}"
    )