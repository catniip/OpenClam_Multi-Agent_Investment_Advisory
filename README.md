# OpenClam: Multi-Agent Investment Advisory

OpenClam is a lightweight multi-agent LLM system for event-driven US equity research. It is designed as a structured research assistant for investment analysis and evaluation, not as an automated trading engine.

The current system combines:

- **News & Macro Agent**: fetches company/news/macro context and reasons about first-order and second-order equity opportunities.
- **Market & Technical Agent**: analyzes price action, momentum, volume, trend, volatility, and relative strength.
- **Fundamental Agent**: analyzes financial statements, earnings-call snippets, guidance, management tone, and thesis impact.
- **CIO Agent**: standardizes agent outputs, summarizes agreement/disagreement, optionally runs one LLM debate round, and produces a final short/medium-term view.

## Project Objective

The core research question is whether a role-specialized multi-agent workflow can produce outputs that are more evidence-grounded, logically consistent, and useful for human decision-making than a single-agent baseline.

The main evaluation focus is **post-earnings trading judgment**: after a company reports earnings and same-day public news becomes available, the agents evaluate the next 7-trading-day and 30-trading-day abnormal return versus QQQ.

## Current Status

- Implemented: News & Macro Agent with Finnhub, NewsAPI, Yahoo Finance, macro proxies, Vertex/OpenAI/Gemini paths, and heuristic fallback.
- Implemented: Fundamental Agent with yfinance financials, optional FMP transcript snippets, robust JSON parsing, and CIO packet conversion.
- Implemented: Market & Technical Agent with technical indicators, LLM reasoning, and CIO packet conversion.
- Implemented: CIO Agent with standardized packet schema, deterministic scoring, optional LLM debate, optional LLM final synthesis, and eval helpers.
- Implemented: Mag7 and expanded AI/Tech Q4 2025 earnings-window evaluation.
- Implemented: categorized output cache so agent outputs can be saved once and reused without repeated model calls.

## Repository Structure

```text
src/openclam/
  agents/
    news_macro/          # News & Macro Agent
    fundamental/         # Fundamental Agent
    market_technical/    # Market & Technical Agent
    cio/                 # CIO synthesis, debate, and eval
  evaluation/
    q4_earnings_cache.py # expanded Q4 2025 universe + cached agent-output workflow
  data/                  # future data connectors and preprocessing
  schemas/               # future shared schemas
  llm/                   # future provider abstractions
  workflow/              # future orchestration
  utils/                 # shared utilities

notebooks/
  01_news_macro_agent_demo.ipynb       # News/Macro demo and standalone eval
  02_market_technical_agent_demo.ipynb # Market/Technical demo
  03_fundamental_agent.ipynb           # Fundamental demo
  04_single_ticker_cio_demo.ipynb      # run all agents + CIO for one ticker
  05_q4_2025_expanded_eval.ipynb       # expanded cached Q4 AI/Tech eval

data/agent_outputs/q4_2025_ai_tech/
  news_macro/          # saved News & Macro outputs
  market_technical/    # saved Market & Technical outputs
  fundamental/         # saved Fundamental outputs
  contexts/            # saved news contexts
  packets/             # standardized CIO-ready packets
  cio/                 # CIO synthesis/debate/final decision outputs
  errors/              # per-ticker errors
  tables/              # universe, price summary, eval tables, aggregate summaries

docs/                  # design and evaluation notes
reports/               # figures and case studies
tests/                 # future tests
```

## Quick Start

### 1. Create Environment

Using venv:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

Using conda:

```bash
conda env create -f environment.yml
conda activate openclam
pip install -e .
```

The package requires Python 3.10+.

### 2. Configure Environment Variables

Create a local `.env` file:

```bash
cp .env.example .env
```

Common variables:

```text
OPENAI_API_KEY=
GEMINI_API_KEY=
FINNHUB_API_KEY=
NEWSAPI_KEY=
FMP_API_KEY=
VERTEX_PROJECT=
VERTEX_LOCATION=us-central1
VERTEX_MODEL=gemini-2.5-flash
OPENAI_MODEL=gpt-5-nano
NEWS_MODEL=gpt-5.4-nano
```

Recommended setup for this project:

- `FINNHUB_API_KEY`: company-news retrieval.
- `NEWSAPI_KEY`: broader historical/news search.
- `VERTEX_PROJECT` + Google Cloud auth: preferred LLM backend for local/Colab school accounts.
- `OPENAI_API_KEY`: optional fallback for OpenAI-compatible runs.
- `FMP_API_KEY`: optional earnings-call transcript snippets for the Fundamental Agent.

### 3. Vertex AI Local Auth

For local Vertex AI use:

```bash
gcloud auth application-default login
```

Then ensure `.env` contains:

```text
VERTEX_PROJECT=your-google-cloud-project-id
VERTEX_LOCATION=us-central1
VERTEX_MODEL=gemini-2.5-flash
```

For Colab:

```python
from google.colab import auth
auth.authenticate_user()
```

## Main Notebooks

### News & Macro Only

```text
notebooks/01_news_macro_agent_demo.ipynb
```

Use this notebook to fetch ticker news context, generate a News/Macro report, and show the standardized CIO packet shape for this agent.

### Single-Ticker Multi-Agent CIO Demo

```text
notebooks/04_single_ticker_cio_demo.ipynb
```

This runs all three agents for one ticker, converts each output into a CIO packet, runs the CIO workflow, and evaluates the final stance against post-earnings abnormal returns.

### Expanded Q4 2025 AI/Tech Eval

```text
notebooks/05_q4_2025_expanded_eval.ipynb
```

This notebook runs a starter universe of 20 AI/Tech companies and saves outputs into the cache folder. The default starter universe includes:

```python
[
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA",
    "AMD", "AVGO", "MU", "TSM", "ASML", "AMAT", "LRCX",
    "ORCL", "DELL", "ANET", "VRT", "CRM", "PLTR",
]
```

Set:

```python
RUN_AGENTS = True
FORCE_RERUN = False
```

The first run generates missing cached outputs. Later runs reuse saved files unless `FORCE_RERUN=True`.

## Evaluation Logic

The current evaluation is **post-earnings**, not strict pre-earnings no-leakage.

Default design:

- Agents may use public news through the earnings date.
- Price anchor is the earnings-date close: `price_anchor="event_close"`.
- Short-term target: next 7 trading days.
- Medium-term target: next 30 trading days.
- Main benchmark: QQQ abnormal return.
- SPY abnormal return is also retained as reference.
- Neutral calls are scored using a configurable neutral band, default `0.02`.

This means the eval asks:

> Given earnings-day information, did the agent/CIO correctly judge whether the stock would outperform or underperform QQQ over the next 7 and 30 trading days?

## CIO Packet Schema

Every agent should expose a CIO-ready packet with this shape:

```python
{
    "ticker": "NVDA",
    "company": "Nvidia",
    "agent_name": "news_macro",
    "short_term_stance": "Bullish",
    "long_term_stance": "Bullish",
    "confidence": 0.75,
    "confidence_rationale": "...",
    "stance_rationale": "...",
    "key_signals": [...],
    "risks": [...],
    "citations": [...],
    "summary": "...",
}
```

Adapters currently available:

```python
news_macro_agent.to_cio_agent_input(news_report)
cio_agent.to_cio_packet_from_market(market_output)
cio_agent.to_cio_packet_from_fundamental(fundamental_output, company="Nvidia")
```

## CIO Debate

The CIO layer can run in two modes:

- Deterministic fallback: transparent weighted aggregation, no model call.
- LLM debate/decision: agents revise their views after seeing peer evidence, then CIO synthesizes final stance.

Example:

```python
cio_eval, cio_results = cio_agent.run_cio_eval(
    summary_df,
    packets_by_ticker,
    long_post_trading_days=30,
    neutral_band=0.02,
    use_llm_debate=True,
    use_llm_decision=True,
    llm_provider="vertex",
    debate_model="gemini-2.5-flash",
    decision_model="gemini-2.5-flash",
    vertex_project=VERTEX_PROJECT,
    vertex_location=VERTEX_LOCATION,
)
```

If debate output says `No LLM debate was run`, the workflow fell back to deterministic debate. Check Vertex/OpenAI credentials.

## Cached Output Workflow

The expanded eval module saves outputs by agent type:

```python
from openclam.evaluation import q4_earnings_cache as q4

earnings_df = q4.q4_2025_ai_tech_earnings_df(["NVDA", "AMD"])
summary_df, paths_df = q4.build_price_tables(earnings_df)

packets_by_ticker, errors_by_ticker = q4.run_q4_2025_universe_agents(
    earnings_df=earnings_df,
    force=False,
    vertex_project=VERTEX_PROJECT,
    vertex_location=VERTEX_LOCATION,
    vertex_model="gemini-2.5-flash",
)

cio_eval, cio_results, cio_summary = q4.run_cached_cio_eval(
    summary_df,
    packets_by_ticker,
    use_llm_debate=True,
    use_llm_decision=True,
    llm_provider="vertex",
    debate_model="gemini-2.5-flash",
    decision_model="gemini-2.5-flash",
    vertex_project=VERTEX_PROJECT,
    vertex_location=VERTEX_LOCATION,
)
```

Cached outputs are stored under:

```text
data/agent_outputs/q4_2025_ai_tech/
```

## Important Notes

- This project is for research and education only.
- Outputs are not financial advice.
- API data may be incomplete, delayed, or noisy.
- Historical earnings dates in the expanded universe are currently hard-coded for stability and should be verified before final reporting.
- QQQ contains many evaluated names, so abnormal return versus QQQ can be diluted by benchmark overlap.
- The current eval does not yet control for beta, options-implied move, factor exposure, or intraday earnings-release timing.

