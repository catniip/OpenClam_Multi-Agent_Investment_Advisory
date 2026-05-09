# Proposal Summary

## Project Goal

OpenClam investigates whether a role-specialized multi-agent LLM system can support event-driven stock insight generation and decision support for US equities.

The system focuses on the post-earnings decision window: after a company reports earnings, can structured agents synthesize public news, market reaction, technical signals, and fundamentals into useful short-term and longer-horizon investment stances?

## Proposed Features and Implementation Status

| Proposed feature | Current implementation |
| --- | --- |
| News and macro event analysis | Implemented in `news_macro_agent.py` with Finnhub, NewsAPI, Yahoo Finance, macro proxies, Vertex/OpenAI/Gemini support, and heuristic fallback. |
| Technical and market analysis | Implemented in `market_technical_agent.py` with trend, momentum, volatility, relative return, and earnings momentum signals. |
| Fundamental analysis | Implemented in `fundamental_agent.py` with financial statement features, valuation metrics, transcript snippets, and structured output parsing. |
| CIO synthesis | Implemented in `cio_agent.py` with packet normalization, conflict detection, optional LLM debate, and final decision synthesis. |
| Evaluation workflow | Implemented in `q4_earnings_cache.py` and `05_q4_2025_expanded_eval.ipynb` for the 55-stock Q4 2025 AI/Tech universe. |
| Reproducible caching | Implemented under `data/agent_outputs/q4_2025_ai_tech/` with per-agent outputs, CIO outputs, errors, and summary tables. |

## Research Question

Does a structured multi-agent CIO workflow provide more stable and useful investment judgment than a single external LLM research pass?

## Main Findings

The current evaluation suggests that the CIO layer provides more balanced multi-horizon performance than external LLM baselines. It does not win every short-term metric, but it achieves stronger long-horizon accuracy and makes agent disagreements visible through the debate workflow.

## Limitations

- Public financial data can be incomplete, delayed, or noisy.
- News and social/market signals can contain substantial noise.
- LLM outputs may still contain unsupported claims.
- Current grounding and consistency checks are diagnostic rather than fully automated metrics.
- The 55-name sample is useful for project evaluation but not large enough to claim production trading validity.

## Future Work

- Add industry-specific expert nodes for Technology, Energy, Healthcare, Financials, and Consumer sectors.
- Add more systematic grounding checks and citation audits.
- Expand the sample across more earnings seasons.
- Fine-tune or prompt-specialize agents on domain-specific financial corpora.
- Calibrate CIO decision rules against validation sets rather than one presentation sample.
