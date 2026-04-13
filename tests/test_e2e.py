import os
import json
from pathlib import Path

from config import settings

os.environ["OPENAI_API_KEY"] = settings.api_key.get_secret_value()

from deepeval import assert_test
from deepeval.metrics import AnswerRelevancyMetric, GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

from agents.research import research


answer_relevancy = AnswerRelevancyMetric(
    threshold=0.7,
    model="gpt-4o-mini",
)

correctness = GEval(
    name="Correctness",
    evaluation_steps=[
        "Check whether the actual output covers the main ideas from the expected output.",
        "Penalize contradiction of important facts or concepts.",
        "Penalize omission of critical details.",
        "Allow different wording if the meaning is preserved.",
    ],
    evaluation_params=[
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
        LLMTestCaseParams.EXPECTED_OUTPUT,
    ],
    model="gpt-4o-mini",
    threshold=0.5,
)


def load_golden_dataset() -> list[dict]:
    dataset_path = Path("tests/golden_dataset.json")

    assert dataset_path.exists(), f"Golden dataset file not found: {dataset_path}"

    data = json.loads(dataset_path.read_text(encoding="utf-8"))

    assert isinstance(data, list), "Golden dataset must be a JSON list."
    assert len(data) > 0, "Golden dataset must not be empty."

    return data


def validate_dataset_item(item: dict, index: int) -> None:
    assert isinstance(item, dict), f"Dataset item at index {index} must be an object."

    required_keys = {"input", "expected_output", "category"}
    missing_keys = required_keys - set(item.keys())

    assert not missing_keys, (
        f"Dataset item at index {index} is missing keys: {sorted(missing_keys)}"
    )

    assert isinstance(item["input"], str) and item["input"].strip(), (
        f"Dataset item at index {index} has invalid 'input'."
    )
    assert isinstance(item["expected_output"], str) and item["expected_output"].strip(), (
        f"Dataset item at index {index} has invalid 'expected_output'."
    )
    assert isinstance(item["category"], str) and item["category"].strip(), (
        f"Dataset item at index {index} has invalid 'category'."
    )


def test_golden_dataset_e2e():
    dataset = load_golden_dataset()

    for index, item in enumerate(dataset):
        validate_dataset_item(item, index)

        user_input = item["input"]
        expected_output = item["expected_output"]

        actual_output = research.invoke({"request": user_input})

        assert isinstance(actual_output, str), (
            f"Research output for dataset item {index} must be a string."
        )
        assert actual_output.strip(), (
            f"Research output for dataset item {index} must not be empty."
        )

        test_case = LLMTestCase(
            input=user_input,
            actual_output=actual_output,
            expected_output=expected_output,
        )

        if item["category"] == "failure_case":
            assert_test(test_case, [correctness])
        else:
            assert_test(test_case, [answer_relevancy, correctness])