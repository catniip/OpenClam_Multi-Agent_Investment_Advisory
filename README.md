# OpenClam: Multi-Agent Investment Advisory

OpenClam is a lightweight multi-agent LLM system for event-driven US equity research. The project is designed as a structured research assistant, not an automated trading engine. It combines role-specialized agents for fundamentals, news and macro, market and technical signals, and a CIO synthesis layer.

## Project Objective

The core research question is whether a role-specialized multi-agent workflow can produce outputs that are more evidence-grounded, logically consistent, and useful for human decision-making than a single-agent baseline.

The first implemented slice is the News & Macro Agent, which collects company news, market news, macro proxy signals, and generates second-order investment opportunity analysis around US-listed equities.

## Current Status

- Implemented: News & Macro Agent prototype
- Implemented: Yahoo Finance, NewsAPI, Finnhub company-news integration
- Implemented: Vertex AI, OpenAI, and Gemini API provider paths
- Placeholder folders: Fundamental Agent, Market & Technical Agent, CIO Agent
- Planned: earnings-window backtesting and multi-agent synthesis

## Repository Structure

```text
src/openclam/
  agents/
    news_macro/          # implemented News & Macro Agent
    fundamental/         # placeholder for Fundamental Agent
    market_technical/    # placeholder for Market & Technical Agent
    cio/                 # placeholder for CIO Agent
  data/                  # future data connectors and preprocessing
  schemas/               # future shared context/report schemas
  llm/                   # future provider and prompt abstractions
  workflow/              # future multi-agent orchestration
  evaluation/            # future backtests and evaluation metrics
  utils/                 # shared utilities

notebooks/               # runnable demos and experiments
docs/                    # design and evaluation notes
data/                    # local raw/processed/output artifacts
reports/                 # case studies and figures
tests/                   # future tests
```

## Quick Start

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the News & Macro demo notebook:

```text
notebooks/01_news_macro_agent_demo.ipynb
```

Existing notebook imports are still supported through the root compatibility link:

```python
from news_macro_agent import collect_context, generate_report, display_report
```

The package path is:

```python
from openclam.agents.news_macro.news_macro_agent import collect_context
```


## Local Development Environment

Option A: Python venv

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

If you plan to use Gemini API or Vertex AI locally, install the optional Gemini/Vertex dependency from PyPI:

```bash
python -m pip install --index-url https://pypi.org/simple -r requirements-llm.txt
# or, after base install:
pip install -e ".[vertex]"
```

If `google-genai` cannot be found, check that pip is using PyPI and that Python is 3.10+:

```bash
python -V
python -m pip config list
python -m pip install --index-url https://pypi.org/simple google-genai
```

Option B: Conda

```bash
conda env create -f environment.yml
conda activate openclam
pip install -e .
# Optional for Gemini API / Vertex AI:
python -m pip install --index-url https://pypi.org/simple -r requirements-llm.txt
```

Create a local `.env` file from the example:

```bash
cp .env.example .env
```

Then fill in any keys you plan to use. At minimum, local runs can work with no keys through the heuristic fallback, but automated news/model analysis is better with Finnhub and Vertex/OpenAI/Gemini configured.

Launch the demo notebook:

```bash
jupyter notebook notebooks/01_news_macro_agent_demo.ipynb
```

## Environment Variables

Create local environment variables based on `.env.example`:

```text
OPENAI_API_KEY=
GEMINI_API_KEY=
FINNHUB_API_KEY=
NEWSAPI_KEY=
VERTEX_PROJECT=
VERTEX_LOCATION=us-central1
```

For Colab with Vertex AI, authenticate first:

```python
from google.colab import auth
auth.authenticate_user()
```

Then call:

```python
report = generate_report(
    context,
    provider="vertex",
    gemini_model="gemini-2.5-flash",
    vertex_project="your-project-id",
    vertex_location="us-central1",
)
```

## Planned Evaluation

The main evaluation track will focus on earnings-event windows:

- collect historical earnings dates
- build event windows around each earnings date
- collect news and macro context automatically
- generate specialist-agent reports
- compare outcomes using post-event returns and abnormal returns versus SPY/QQQ
- compare multi-agent output against single-agent and no-debate baselines
