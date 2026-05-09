# Evaluation Plan

The main experiment evaluates whether a structured multi-agent CIO workflow can improve post-earnings investment judgment relative to external single-LLM baselines.

## Task Definition

For each company in the Q4 2025 AI/Tech earnings universe, the system receives:

- ticker and company name,
- earnings date,
- event-window news and market context,
- available financial and technical data.

The system predicts:

- **Short-term stance**: Bullish / Neutral / Bearish over the next 7 trading days.
- **Long-term stance**: Bullish / Neutral / Bearish over the next 30 trading days.

The target is abnormal return versus QQQ. A bullish stance expects outperformance, a bearish stance expects underperformance, and a neutral stance expects the abnormal return to remain inside the configured neutral band.

## Leakage Constraint

The post-earnings setup allows information available by the end of the company's earnings date, including:

- earnings release,
- same-day news,
- same-day market reaction,
- contemporaneous analyst or market commentary.

The setup excludes later news, later revisions, and realized 7-day or 30-day returns when assigning labels.

## Metrics

### Usefulness

Directional accuracy of Bullish / Neutral / Bearish stances against realized abnormal returns versus QQQ over 7-day and 30-day horizons.

### Grounding

Whether rationales can be traced to cached news, earnings data, technical signals, market data, and citations. This is currently assessed as a qualitative diagnostic rather than a fully automated citation-accuracy metric.

### Consistency

Whether the CIO layer identifies disagreements among agents and produces coherent short-term versus long-term views.

## Baselines

The primary external baselines are:

- GPT-5
- Gemini 2.5 Pro
- Claude Sonnet 4.6

Each external baseline receives the same universe, dates, leakage rules, horizon definitions, and label format.

Internal diagnostics include:

- individual agent outputs,
- no-debate raw aggregation,
- uniform-weight aggregation,
- alternative weight profiles.

These internal diagnostics help explain performance but are not the main external benchmark.

## Reproducibility

The expanded evaluation notebook writes all intermediate artifacts to `data/agent_outputs/q4_2025_ai_tech/`. To reproduce the cached evaluation:

1. Install dependencies and configure `.env`.
2. Run `notebooks/05_q4_2025_expanded_eval.ipynb`.
3. Keep `FORCE_RERUN=False` to reuse completed cached outputs.
4. Set `FORCE_RERUN=True` only when intentionally regenerating agent outputs.

Important output tables:

- `tables/cio_eval_full_55.csv`
- `tables/agent_strategy_summary_with_external_baselines.csv`
- `tables/agent_strategy_bucket_summary_with_external_baselines.csv`
- `tables/external_baselines_vs_cio.csv`
