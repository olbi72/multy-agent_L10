# Multi-Agent Research System with Automated Testing

## Overview

This project is a multi-agent research system built with **LangChain**, **LangGraph**, and **OpenAI models**.

The system was originally developed as a research workflow in homework 8 and then extended in homework 10 with an automated testing layer using **DeepEval** and **pytest**.

The project contains:

- a **Planner agent** that creates a structured research plan
- a **Researcher agent** that gathers and synthesizes information
- a **Critic agent** that evaluates the research output
- a **Supervisor agent** that coordinates the full workflow
- a **local knowledge base** with hybrid retrieval
- an **automated test suite** for component-level, tool-level, and end-to-end evaluation

---

## Main Goal

The goal of the system is to answer research-style user requests through a multi-step agent workflow instead of relying on a single direct LLM response.

The goal of the testing layer is to provide a **repeatable baseline** for quality evaluation, instead of relying only on manual “vibe checks”.

---

## Architecture

The system follows this high-level flow:

```text
User Request
   ↓
Planner
   ↓
Researcher
   ↓
Critic
   ↓
Supervisor decision:
   - APPROVE → save report
   - REVISE → send revision requests back to Researcher

Agent Roles
1. Planner

The Planner analyzes the user request and converts it into a structured ResearchPlan.

It decides:

what the real research goal is
which search queries should be used
which sources should be checked
what the expected final output format should look like
2. Researcher

The Researcher executes the plan.

It can use:

web search
URL reading
local knowledge base retrieval

Its task is to gather evidence and produce a structured research summary in markdown form.

3. Critic

The Critic evaluates the Researcher’s findings.

It checks:

completeness
grounding
structure
freshness where relevant
whether revisions are needed

It returns a structured CritiqueResult with either:

APPROVE
REVISE
4. Supervisor

The Supervisor coordinates the whole process.

It:

calls the Planner first
sends the plan to the Researcher
sends the findings to the Critic
triggers another research round if revision is needed
saves the final report after approval

multy-agent_L10/
├── agents/
│   ├── __init__.py
│   ├── critic.py
│   ├── planner.py
│   └── research.py
├── data/
│   ├── langchain.pdf
│   ├── large-language-model.pdf
│   └── retrieval-augmented-generation.pdf
├── tests/
│   ├── __init__.py
│   ├── golden_dataset.json
│   ├── test_critic.py
│   ├── test_e2e.py
│   ├── test_planner.py
│   ├── test_researcher.py
│   └── test_tools.py
├── .gitignore
├── config.py
├── ingest.py
├── main.py
├── requirements.txt
├── retriever.py
├── schemas.py
├── supervisor.py
└── tools.py

File-by-File Explanation
config.py

Contains:

project settings
model configuration
directory paths
retrieval parameters
agent system prompts

This file is the main configuration center of the project.

It defines:

the OpenAI model name
embedding model
search limits
chunk size and overlap
reranker model
max iterations
system prompts for Planner, Researcher, Critic, and Supervisor
schemas.py

Contains Pydantic schemas used for structured outputs.

Main schemas:

ResearchPlan
CritiqueResult
LocalCritiqueResult

These schemas help ensure that agent outputs are structured and easier to test.

tools.py

Contains tools used by the agents.

Implemented tools:

web_search
read_url
knowledge_search
save_report
write_report (backward-compatible alias)

This module also contains low-level helper functions:

web_search_raw
read_url_raw
save_report_raw
retriever.py

Implements hybrid retrieval over the local knowledge base.

Main steps:

load FAISS vector store
load BM25 chunks
perform semantic retrieval
perform BM25 retrieval
deduplicate documents
rerank documents with a cross-encoder
return the top final results

This module is responsible for knowledge_search.

ingest.py

Builds the local knowledge base from files in the data/ directory.

It:

loads PDF, TXT, and MD files
splits them into chunks
creates embeddings
saves the FAISS index
saves serialized chunks for BM25 retrieval

This file should be run when the document collection changes.

agents/planner.py

Defines the Planner agent and the plan tool.

The Planner:

takes a user request
optionally performs light exploration
returns a structured ResearchPlan
agents/research.py

Defines the Researcher agent and the research tool.

The Researcher:

executes research based on the plan
uses web search, URL reading, and/or local knowledge search
returns research findings in markdown format
agents/critic.py

Defines the Critic agent and the critique tool.

The Critic:

evaluates findings
may perform light verification
returns a structured critique result
decides whether the answer is ready or must be revised
supervisor.py

Defines the Supervisor agent.

The Supervisor:

orchestrates the workflow
starts with planning
calls the Researcher
calls the Critic
loops if revisions are required
saves the final report after approval

It also includes human-in-the-loop control for save actions.

main.py

Provides the interactive command-line interface for the system.

It supports:

running the Supervisor
showing tool calls
handling approval/edit/reject steps
resuming after interrupts

This is the main entry point for interactive use.

Local Knowledge Base

The project uses a hybrid retriever that combines:

semantic search through FAISS embeddings
keyword search through BM25
cross-encoder reranking

This improves retrieval quality compared to using only one retrieval strategy.

Retrieval Flow
User query
   ↓
Semantic retrieval (FAISS)
   +
BM25 retrieval
   ↓
Deduplication
   ↓
Cross-encoder reranking
   ↓
Top-k final passages
Installation
1. Clone the repository
git clone https://github.com/olbi72/multy-agent_L10.git
cd multy-agent_L10
2. Create virtual environment
python -m venv .venv
3. Activate virtual environment
Windows PowerShell
.venv\Scripts\Activate.ps1
Windows CMD
.venv\Scripts\activate
4. Install dependencies
pip install -r requirements.txt
Environment Variables

Create a .env file in the project root.

Example:

api_key=YOUR_OPENAI_API_KEY
OPENAI_API_KEY=YOUR_OPENAI_API_KEY
model_name=gpt-4o-mini
embedding_model=text-embedding-3-small
Why both api_key and OPENAI_API_KEY?
api_key is used by the project’s internal configuration via pydantic_settings
OPENAI_API_KEY is needed by DeepEval metrics that directly access OpenAI
Preparing the Knowledge Base

Before using the local retrieval pipeline, build the vector store.

Run:

python ingest.py

This will:

load documents from data/
split them into chunks
create embeddings
save the index to vector_store/
Running the Project

Run the interactive system:

python main.py

The CLI supports:

user requests
tool execution display
approval flow for saving reports
revision loops after critique
Automated Testing

The project includes a full testing layer in tests/.

Test Types
1. Golden Dataset

tests/golden_dataset.json

Contains a manually reviewed set of examples:

happy path
edge cases
failure cases

Each item has:

input
expected_output
category
2. Component Tests
tests/test_planner.py

Tests whether the Planner:

produces a valid structured plan
creates meaningful search queries
chooses relevant sources
tests/test_researcher.py

Tests whether the Researcher:

produces grounded output
returns non-empty results
tests/test_critic.py

Tests whether the Critic:

produces actionable critique
returns structured evaluation output
3. Tool Correctness Tests
tests/test_tools.py

Checks expected tool usage behavior for:

Planner
Researcher
Supervisor

These tests are simplified behavioral tests rather than full runtime trace tests.

4. End-to-End Test
tests/test_e2e.py

Runs evaluation over the golden dataset.

It checks:

answer relevancy for normal cases
correctness against expected output
category-aware evaluation for failure cases
Running Tests
Run all tests
python -m pytest tests -q
Run a single test file
python -m pytest tests/test_planner.py -q
python -m pytest tests/test_researcher.py -q
python -m pytest tests/test_critic.py -q
python -m pytest tests/test_tools.py -q
python -m pytest tests/test_e2e.py -q
Test Results Baseline


Metrics Used

The project uses DeepEval metrics such as:

GEval
AnswerRelevancyMetric
ToolCorrectnessMetric
Custom evaluation logic included in tests:
plan quality
critique quality
groundedness
correctness
tool correctness
Design Decisions
Why automated testing was added

The original system from homework 8 relied mainly on manual quality inspection.

Homework 10 adds:

repeatable evaluation
regression-style testing
structured baseline measurement
better visibility into system quality
Why some thresholds are moderate

Thresholds were intentionally set to realistic baseline levels rather than overly strict values.

This reflects the actual purpose of early evaluation:

establish a working baseline
improve the system incrementally over time
Why failure cases were calibrated

Some failure cases were adjusted so that they are:

stable for automatic evaluation
aligned with the actual behavior of the current system
suitable for baseline testing

This does not mean the system is perfect.
It means the dataset was calibrated to serve as a reliable automated benchmark.

Known Limitations
Tool correctness tests are simplified and do not yet use full runtime traces
End-to-end evaluation currently runs through the Researcher-centered flow rather than the full Supervisor orchestration for every dataset item
Some web pages may return 403 Forbidden, which can reduce external evidence quality
Some safety or subjective failure cases are harder to evaluate automatically with generic relevancy metrics
Full test execution is relatively slow because DeepEval metrics call LLMs
Possible Future Improvements
Add full runtime trace capture for tool correctness
Add dedicated refusal/safety metrics for failure cases
Extend end-to-end tests to full Supervisor pipeline
Add reporting of per-case metric summaries
Separate web-dependent and KB-only tests
Add CI integration for automated GitHub test runs
Technologies Used
Python
LangChain
LangGraph
OpenAI
FAISS
BM25
sentence-transformers
DeepEval
pytest
trafilatura
DDGS
Example Use Cases
compare RAG strategies
explain retrieval methods
summarize concepts from local ingested documents
build structured research reports
evaluate multi-agent research quality automatically


