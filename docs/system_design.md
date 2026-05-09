# System Design

OpenClam is organized as a modular multi-agent research workflow for post-earnings US equity analysis. The system is intentionally split into specialist agents so that each model call has a narrow responsibility and the CIO layer can compare conflicting evidence.

## Architecture

```text
Ticker + earnings date
        |
        v
Event-window data collection
        |
        +--> News & Macro Agent
        +--> Market & Technical Agent
        +--> Fundamental Agent
        |
        v
Standardized CIO packets
        |
        v
CIO synthesis + optional debate
        |
        v
Final short-term and long-term stance
        |
        v
Evaluation versus abnormal returns vs QQQ
```

## Agent Responsibilities

- **News & Macro Agent** collects company-specific news, related market context, and macro proxies. It produces short-term and long-term stances with a news-driven rationale.
- **Market & Technical Agent** analyzes price action, trend, momentum, volatility, volume, benchmark-relative movement, and post-earnings momentum signals.
- **Fundamental Agent** analyzes company financials, growth, profitability, valuation, available transcript snippets, and earnings-quality signals.
- **CIO Agent** normalizes agent outputs into a shared packet schema, identifies agreement and disagreement, optionally runs a debate round, and returns the final investment view.

## Data Flow and Caching

Expanded Q4 2025 experiments use `src/openclam/evaluation/q4_earnings_cache.py` as the orchestration layer. Outputs are cached under:

```text
data/agent_outputs/q4_2025_ai_tech/
```

The cache is separated by output type:

- `contexts/`: fetched news and macro contexts.
- `news_macro/`, `market_technical/`, `fundamental/`: agent-specific outputs.
- `packets/`: standardized CIO-ready packet lists.
- `cio/`: final CIO synthesis, debate responses, and final decisions.
- `tables/`: price tables, evaluation tables, and aggregate summaries.

This lets notebooks rerun analysis without repeatedly calling APIs or LLMs.

## Error Handling

The workflow is designed to keep partial results:

- Per-ticker errors are written to `errors/`.
- Missing API keys skip optional sources instead of crashing the full run.
- LLM failures fall back to deterministic or heuristic outputs where possible.
- Cached runs can resume missing tickers without overwriting completed files unless `force=True`.

## Optimization Choices

- Cached agent outputs avoid repeated LLM calls.
- The expanded evaluation can run incrementally by ticker.
- Price tables and CIO tables are persisted as CSV/JSON for reproducibility.
- Raw aggregation baselines are computed directly from cached packets without new model calls.
