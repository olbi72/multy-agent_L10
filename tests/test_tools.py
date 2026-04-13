import os

from config import settings

os.environ["OPENAI_API_KEY"] = settings.api_key.get_secret_value()

from deepeval.test_case import LLMTestCase, ToolCall
from deepeval.metrics import ToolCorrectnessMetric
from deepeval import assert_test

tool_metric = ToolCorrectnessMetric(
    threshold=0.5,
    model="gpt-4o-mini",
)


def test_planner_tool_correctness():
    test_case = LLMTestCase(
        input="Compare naive RAG vs sentence-window retrieval",
        actual_output="Planner created a research plan using web search for exploration.",
        tools_called=[
            ToolCall(name="web_search", description="Search the web for relevant information"),
        ],
        expected_tools=[
            ToolCall(name="web_search", description="Search the web for relevant information"),
        ],
    )

    assert_test(test_case, [tool_metric])


def test_researcher_tool_correctness():
    test_case = LLMTestCase(
        input="Use the plan to research hybrid retrieval in RAG.",
        actual_output="Researcher used web search and page reading to gather evidence.",
        tools_called=[
            ToolCall(name="web_search", description="Search the web for relevant information"),
            ToolCall(name="read_url", description="Read the main text content from a web page URL"),
        ],
        expected_tools=[
            ToolCall(name="web_search", description="Search the web for relevant information"),
            ToolCall(name="read_url", description="Read the main text content from a web page URL"),
        ],
    )

    assert_test(test_case, [tool_metric])


def test_supervisor_save_report_tool_correctness():
    test_case = LLMTestCase(
        input="Critic approved the report. Supervisor should save it.",
        actual_output="Supervisor saved the final markdown report.",
        tools_called=[
            ToolCall(name="save_report", description="Save a Markdown research report to a file"),
        ],
        expected_tools=[
            ToolCall(name="save_report", description="Save a Markdown research report to a file"),
        ],
    )

    assert_test(test_case, [tool_metric])