# News & Macro Agent Prototype

This is the first runnable slice of the multi-agent stock insight system. It focuses on the News & Macro Agent described in the project design document, using an event-driven second-order effects prompt.

## Quick start in Jupyter

1. Open `news_macro_agent_demo.ipynb`.
2. Run the first cell.
3. Change `TICKER`, `COMPANY`, and `EVENT_DATE`.
4. Run the remaining cells.

The notebook works offline with sample data. For live data:

```bash
pip install yfinance requests openai google-genai
```

Optional environment variables:

```bash
export NEWSAPI_KEY="..."
export OPENAI_API_KEY="..."
export GEMINI_API_KEY="..."
```

Provider order is `OPENAI_API_KEY` first, then `GEMINI_API_KEY`, then Vertex AI if `vertex_project` is passed, then a transparent heuristic fallback. With any model provider, the agent asks for a structured English JSON report using the event-driven hedge fund manager prompt.

## Direct pasted-news workflow

In Colab, the easiest path is:

```python
from news_macro_agent import analyze_news_text, display_report

NEWS_TEXT = """
Paste one full news article or several news bullets here.
"""

report = analyze_news_text(
    NEWS_TEXT,
    ticker="MSFT",
    company="Microsoft",
    event_date="2026-04-26",
    provider="auto",
)

display_report(report)
```

## Latest vs historical case studies

For current news:

```python
context = collect_context(
    ticker="NVDA",
    company="Nvidia",
    news_mode="latest",
    lookback_days=14,
)
```

For a historical event window:

```python
context = collect_context(
    ticker="NVDA",
    company="Nvidia",
    event_date="2025-04-26",
    news_mode="event_window",
    lookback_days=14,
)
```

`latest` keeps `event_date` only as metadata and fetches current news. `event_window` filters available sources to `event_date - lookback_days` through `event_date + 1 day`. Yahoo Finance usually returns current news only, so historical case studies work best with NewsAPI, manually pasted `NEWS_TEXT`, or another historical news source.

To force Gemini in Colab:

```python
import os
os.environ["GEMINI_API_KEY"] = "..."

report = analyze_news_text(
    NEWS_TEXT,
    ticker="MSFT",
    company="Microsoft",
    provider="gemini",
    gemini_model="gemini-2.5-flash",
)
```

To use Vertex AI in Colab with your school Google account:

```python
from google.colab import auth
auth.authenticate_user()

report = analyze_news_text(
    NEWS_TEXT,
    ticker="MSFT",
    company="Microsoft",
    provider="vertex",
    gemini_model="gemini-2.5-flash",
    vertex_project="csee4121-488316",
    vertex_location="us-central1",
)
```

## Output schema

The agent returns English fields:

- `news_summary`
- `core_insight`
- `mainstream_stocks`
- `opportunities`
- `risk_notes`
- `citations`
- `raw_context`

This matches the project plan: each specialist agent should produce a structured report that the CIO agent can later synthesize and challenge.
